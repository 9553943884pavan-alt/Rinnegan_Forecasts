"""Provenance labels for the published pack.

A source entry in a published sidecar says what a file is. It does not say where it sat. Three
label schemes carry that, and none of them names a host, an account or a directory:

``feed-capture:<source>/<rel>``
    one raw file the production feed's ``<source>`` collector captured, ``<rel>`` being the
    asset and capture day it was filed under.
``feed-db:<table>``
    one table dump taken from the feed's database.
``build:<lane>/<rel>``
    an artifact one of this dataset's own build lanes produced.

The gate that applies the convention to what the build lanes wrote, and refuses an assembly that
still carries an address, is ``pack.scrub``, which runs on the organizers' side only.
"""

from __future__ import annotations

from pathlib import PurePosixPath

CAPTURE = "feed-capture"
DB = "feed-db"
BUILD = "build"


def capture_label(source: str, rel: str) -> str:
    """``feed-capture:<source>/<rel>`` for one raw file of the feed's ``<source>`` collector."""
    return f"{CAPTURE}:{source}/{PurePosixPath(rel).as_posix()}"


def db_label(table: str) -> str:
    """``feed-db:<table>`` for one table dump."""
    return f"{DB}:{table}"


def build_label(path: object) -> str:
    """``build:<lane>/<rel>`` for a build-lane artifact, by the structure of its path."""
    return label_path(str(path))


def label_path(path: str) -> str:
    """The label for an absolute path, read off the path's own structure.

    ``.../raw/{feed,vps}/pg/<table>.csv.gz`` is a table dump and ``.../raw/{feed,vps}/<source>/<rel>``
    a capture -- both spellings, because sidecars written before the directory was renamed are
    still read -- ``.../build/<lane>/<rel>`` a lane artifact,
    ``.../build/<lane>/<rel>`` a lane artifact, and a path inside the package is kept from
    ``swing_dataset/`` on so a recorded command still says which module ran. Anything else keeps
    its file name alone.
    """
    pure = PurePosixPath(path)
    parts = pure.parts
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        rest = parts[i + 1 :]
        if not rest:
            continue
        if part in ("vps", "feed"):
            if rest[0] == "pg":
                return db_label(PurePosixPath(rest[-1]).name.split(".", 1)[0]) if len(rest) > 1 else f"{DB}:"
            return capture_label(rest[0], "/".join(rest[1:])) if len(rest) > 1 else f"{CAPTURE}:{rest[0]}"
        if part == "build":
            return f"{BUILD}:" + "/".join(rest)
        if part == "swing_dataset":
            return "swing_dataset/" + "/".join(rest)
    return pure.name
