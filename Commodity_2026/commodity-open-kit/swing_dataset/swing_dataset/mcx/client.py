"""Polite, captured HTTP for mcxindia.com.

The MCX site sits behind an Akamai edge that answers 403 to plain HTTP
clients (httpx with a browser user agent included; measured 2026-09-15).
``curl_cffi`` with Chrome impersonation is answered 200, the same transport
the production feed uses for the MCX circular listing. Every response body is
stored as received under ``build/mcx/raw/`` and indexed in ``INDEX.csv``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import common
from . import LANE

ORIGIN = "https://www.mcxindia.com"
MIN_INTERVAL_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 60.0
RETRY_BACKOFF_SECONDS = (2.0, 4.0, 8.0)


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int | None, detail: str) -> None:
        self.url = url
        self.status = status
        self.detail = detail
        super().__init__(f"{detail} ({url}, status={status})")


@dataclass(slots=True)
class Capture:
    path: Path
    url: str
    fetched_at: datetime
    status: int
    body: bytes
    content_type: str | None

    @property
    def sha256(self) -> str:
        return common.sha256_bytes(self.body)


@dataclass
class McxClient:
    """One session against the MCX edge, rate-limited to one request per second."""

    timeout: float = DEFAULT_TIMEOUT_SECONDS
    min_interval: float = MIN_INTERVAL_SECONDS
    lane: str = LANE
    _session: object = field(default=None, init=False, repr=False)
    _last_request: float = field(default=0.0, init=False, repr=False)

    def _get_session(self):
        if self._session is None:
            from curl_cffi import requests as curl_requests

            self._session = curl_requests.Session(impersonate="chrome")
        return self._session

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        accept_statuses: tuple[int, ...] = (200,),
    ) -> tuple[int, bytes, str | None, datetime]:
        """GET with retries and backoff; returns (status, body, content-type, fetched_at)."""
        session = self._get_session()
        shaped = {"Accept-Language": "en-US,en;q=0.9", **(headers or {})}
        last: FetchError | None = None
        for attempt, backoff in enumerate((*RETRY_BACKOFF_SECONDS, None)):
            self._throttle()
            fetched_at = common.now_utc()
            try:
                response = session.get(url, headers=shaped, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 - curl_cffi raises its own hierarchy
                last = FetchError(url, None, f"transport error: {type(exc).__name__}: {exc}")
            else:
                status = int(response.status_code)
                if status in accept_statuses:
                    return (
                        status,
                        response.content,
                        response.headers.get("content-type"),
                        fetched_at,
                    )
                last = FetchError(url, status, f"http {status} on attempt {attempt + 1}")
                if status == 403:
                    last = FetchError(url, status, "blocked by the edge (403)")
            if backoff is None:
                break
            time.sleep(backoff)
        assert last is not None
        raise last

    def capture(
        self,
        url: str,
        relative_path: str,
        *,
        headers: dict[str, str] | None = None,
        accept_statuses: tuple[int, ...] = (200,),
    ) -> Capture:
        """GET ``url`` and store the bytes at ``build/<lane>/raw/<relative_path>``.

        The file is overwritten with the newest bytes; the INDEX keeps every
        fetch as its own row, so nothing is lost.
        """
        status, body, content_type, fetched_at = self.get(
            url, headers=headers, accept_statuses=accept_statuses
        )
        path = common.lane_root(self.lane) / "raw" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        common.record_capture(self.lane, path, url, fetched_at, body, status)
        return Capture(path, url, fetched_at, status, body, content_type)
