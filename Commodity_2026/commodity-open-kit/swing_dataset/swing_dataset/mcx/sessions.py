"""MCX session calendar 2021-2026 from the exchange's own circulars.

Every date, time and state below is stated by a circular captured under
``build/mcx/raw/circulars/<id>.pdf`` (see ``CIRCULARS``). The encoding follows
the upstream collector's calendar: the annual *Trading Holidays*
circular gives each holiday's morning/evening state, and the *Revision in
Trading Hours* circulars give the evening close (23:30 IST while New York is on
daylight time, 23:55 IST otherwise). A close is only asserted for dates a
retained circular covers; later dates fall back to the shorter 23:30 close with
``hours_notified=false``, exactly as the feed does.

Hours are those for internationally referenceable non-agricultural commodities
(the four products in this dataset). Agricultural contracts close earlier.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa

from .. import common
from . import LANE
from .client import McxClient

FIRST_DATE = date(2021, 1, 1)
LAST_DATE = date(2026, 12, 31)
MORNING_OPEN = time(9, 0)
MORNING_CLOSE = time(17, 0)
EVENING_OPEN = time(17, 0)
CLOSE_DST = time(23, 30)
CLOSE_STANDARD = time(23, 55)
FALLBACK_CLOSE = CLOSE_DST
IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class Circular:
    id: str
    number: str
    issued: date
    title: str
    url: str


#: Every circular the calendar relies on. Numbers are as printed in the PDF.
CIRCULARS: tuple[Circular, ...] = (
    Circular(
        "mcx-circular-763-2020-trading-hours",
        "MCX/TRD/763/2020",
        date(2020, 10, 13),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2020/october/circular-763---2020612a064757fb64e3bdfdff00007acb35guab8e7f08-f25f-4529-8da1-882e6137eede.pdf?sfvrsn=ad224966_0",
    ),
    Circular(
        "mcx-circular-931-2020-trading-holidays-2021",
        "MCX/TRD/931/2020",
        date(2020, 12, 11),
        "Trading Holidays for the Calendar Year 2021",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2020/december/circular-931---2020gu38ca59cd-7915-43b4-a1ff-b082d7d5a5d1.pdf?sfvrsn=11e9170_0",
    ),
    Circular(
        "mcx-circular-085-2021-trading-hours",
        "MCX/TRD/085/2021",
        date(2021, 2, 15),
        "Revision in Trading Timings",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2021/february/circular-085-2021gu0b8fae08-251c-4833-bd69-93f4d05cdf70.pdf?sfvrsn=69d3ab8d_0",
    ),
    Circular(
        "mcx-circular-646-2021-trading-hours",
        "MCX/TRD/646/2021",
        date(2021, 10, 8),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2021/october/circular-646-2021gu6e709d40-df2f-41a3-a9a3-aa19cad3962d.pdf?sfvrsn=d3d0907e_0",
    ),
    Circular(
        "mcx-circular-671-2021-muhurat-2021-11-04",
        "MCX/TRD/671/2021",
        date(2021, 10, 21),
        "Muhurat Trading Session on account of Diwali on November 04, 2021",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2021/october/circular-671-2021gu4042631c-f292-4004-bcf5-1b493bf5b91c.pdf?sfvrsn=b13c08b8_0",
    ),
    Circular(
        "mcx-circular-764-2021-trading-holidays-2022",
        "MCX/TRD/764/2021",
        date(2021, 12, 10),
        "Trading Holidays for the Calendar Year 2022",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2021/december/circular-764-2021gubc959d92-1933-48a2-bec0-872344823ea9.pdf?sfvrsn=d637d37a_0",
    ),
    Circular(
        "mcx-circular-076-2022-trading-hours",
        "MCX/TRD/076/2022",
        date(2022, 2, 14),
        "Revision in Trading Timings",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2022/february/circular-076-2022gu2341741e-d5b1-4c17-88e4-3c15c30cd131.pdf?sfvrsn=4635184_0",
    ),
    Circular(
        "mcx-circular-574-2022-trading-hours",
        "MCX/TRD/574/2022",
        date(2022, 10, 7),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2022/october/circular-574-2022guf769761e-6856-4063-9eaa-1ec6dd7c5921.pdf?sfvrsn=d8496884_0",
    ),
    Circular(
        "mcx-circular-586-2022-muhurat-2022-10-24",
        "MCX/TRD/586/2022",
        date(2022, 10, 11),
        "Muhurat Trading Session on account of Diwali on October 24, 2022",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2022/october/circular-586-2022gu36ebc63c-ddc9-45a6-bf5a-d9b670ee2a6a.pdf?sfvrsn=52a37e2b_0",
    ),
    Circular(
        "mcx-circular-710-2022-trading-holidays-2023",
        "MCX/TRD/710/2022",
        date(2022, 12, 9),
        "Trading Holidays for the Calendar Year 2023",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2022/december/circular---710-2022gu3c04e8dd-2eb5-49e2-901f-c27813a3fc06.pdf?sfvrsn=a0f055f5_0",
    ),
    Circular(
        "mcx-circular-095-2023-trading-hours",
        "MCX/TRD/095/2023",
        date(2023, 2, 13),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2023/february/circular---095-2023gufec3679d-b9de-4fd3-b8a6-42f71762a998.pdf?sfvrsn=70a41003_0",
    ),
    Circular(
        "mcx-circular-409-2023-holiday-revision-bakri-id",
        "MCX/TRD/409/2023",
        date(2023, 6, 27),
        "Revision in Trading Holiday on account of Bakri ID",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2023/june/circular-409-2023gu40bf789b-ae02-4fad-9653-995e0d8f0061.pdf?sfvrsn=24472ae1_0",
    ),
    Circular(
        "mcx-circular-675-2023-trading-hours",
        "MCX/TRD/675/2023",
        date(2023, 10, 6),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2023/october/circular-675-2023gu25570a47-dbf8-4cff-a883-e8e7d0085379.pdf?sfvrsn=26662f0b_0",
    ),
    Circular(
        "mcx-circular-700-2023-trading-hours-2023-10-16",
        "MCX/TRD/700/2023",
        date(2023, 10, 12),
        "Revision in Trading Hours for October 16, 2023",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2023/october/circuar-700-2023guedcccf74-865d-4ba0-a6cf-ce627f613283.pdf?sfvrsn=766704e9_0",
    ),
    Circular(
        "mcx-circular-740-2023-muhurat-2023-11-12",
        "MCX/TRD/740/2023",
        date(2023, 10, 27),
        "Muhurat Trading Session on account of Diwali on November 12, 2023",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2023/october/circular-740-2023gu594fb6fb-c14b-40db-b435-3df23d59555f.pdf?sfvrsn=26d139bc_0",
    ),
    Circular(
        "mcx-circular-872-2023-trading-holidays-2024",
        "MCX/TRD/872/2023",
        date(2023, 12, 14),
        "Trading Holidays for the Calendar Year 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2023/december/circular-872-2023gu79208045-e0ef-481b-89bf-d89619b4fba8.pdf?sfvrsn=7b65d09a_0",
    ),
    Circular(
        "mcx-circular-037-2024-holiday-2024-01-22",
        "MCX/TRD/037/2024",
        date(2024, 1, 19),
        "Holiday declared by Government of Maharashtra on January 22, 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/january/circular-037-2024b0d4024857fb64e3bdfdff00007acb35gucbc618a7-b38f-42d1-b83d-73e86e71a6a0.pdf?sfvrsn=a7034672_0",
    ),
    Circular(
        "mcx-circular-082-2024-trading-hours",
        "MCX/TRD/082/2024",
        date(2024, 2, 9),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/february/circular-082-2024gu52b563c7-7318-4327-8c2b-9c64889a8151.pdf?sfvrsn=feab49aa_0",
    ),
    Circular(
        "mcx-circular-087-2024-trading-hours-2024-02-13",
        "MCX/CSQ/087/2024",
        date(2024, 2, 13),
        "Revision in Trading Hours for 13th Feb 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/february/mcxcsq0872024-revision-in-trading-hours-for-13the-feb-2024gu9ba26e02-d2c6-4656-8cbe-313d9849a25c.pdf?sfvrsn=cfadbfab_0",
    ),
    Circular(
        "mcx-circular-088-2024-trading-hours-2024-02-13",
        "MCX/CSQ/088/2024",
        date(2024, 2, 13),
        "Revision in Trading Hours February 13, 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/february/mcxcsq0882024-revision-in-trading-hours-for-13the-feb-2024gu143a9f70-ebc0-418d-9e9c-d5e0f1b48d03.pdf?sfvrsn=c0ba4da7_0",
    ),
    Circular(
        "mcx-circular-089-2024-trading-hours-2024-02-13",
        "MCX/CSQ/89/2024",
        date(2024, 2, 13),
        "Revision in Trading Hours February 13, 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/february/mcxcsq0892024-revision-in-trading-hours-for-13the-feb-2024-03gu5769f3ea-885f-43b0-9849-b75ca9bcdacb.pdf?sfvrsn=d7ae2e67_0",
    ),
    Circular(
        "mcx-circular-219-2024-holiday-2024-05-20",
        "MCX/TRD/219/2024",
        date(2024, 4, 9),
        "Trading Holiday on account of General Parliamentary Elections 2024 on May 20 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/april/circular-219-2024gub46cefe8-daac-472a-ba37-68fcd43db046.pdf?sfvrsn=b2716a6f_0",
    ),
    Circular(
        "mcx-circular-643-2024-trading-hours",
        "MCX/TRD/643/2024",
        date(2024, 9, 27),
        "Revision in trading hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/september/circular-643-2024gu941bf9ae-b996-4d5c-9f2b-090896ece6c0.pdf?sfvrsn=2c2bd5d3_0",
    ),
    Circular(
        "mcx-circular-704-2024-muhurat-2024-11-01",
        "MCX/TRD/704/2024",
        date(2024, 10, 21),
        "Muhurat Trading Session on account of Diwali on November 01, 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/october/circular-704-2024gufb868326-a938-42d9-aea7-13475772b26d.pdf?sfvrsn=bf7da1bb_0",
    ),
    Circular(
        "mcx-circular-738-2024-holiday-2024-11-20",
        "MCX/TRD/738/2024",
        date(2024, 11, 8),
        "Trading Holiday on account of Maharashtra Legislative Assembly General Election on November 20, 2024",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/november/circular-738-2024gubc8bbce8-0732-46e4-a7aa-1d72d41a4eeb.pdf?sfvrsn=9e334992_0",
    ),
    Circular(
        "mcx-circular-814-2024-trading-holidays-2025",
        "MCX/TRD/814/2024",
        date(2024, 12, 13),
        "Trading Holidays for the Calendar Year 2025",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/december/circular-814-2024gub7919944-a07e-4e62-b440-51fdece1c579.pdf?sfvrsn=c7593ddd_0",
    ),
    Circular(
        "mcx-circular-841-2024-live-session-2025-02-01",
        "MCX/TRD/841/2024",
        date(2024, 12, 24),
        "Live Trading Session on February 01, 2025 (Saturday)",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2024/december/circular---841-2024gubcdb6400-80b9-4cf9-9b72-885e9a19bbec.pdf?sfvrsn=154e71d8_0",
    ),
    Circular(
        "mcx-circular-071-2025-trading-hours",
        "MCX/TRD/071/2025",
        date(2025, 2, 7),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2025/february/circular-071-2025gue247f267-0cab-4e56-a006-0f9e76d0fe20.pdf?sfvrsn=2b327666_0",
    ),
    Circular(
        "mcx-circular-477-2025-muhurat-2025-10-21",
        "MCX/TRD/477/2025",
        date(2025, 9, 23),
        "Muhurat Trading Session on account of Diwali on October 21, 2025",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2025/september/circular-477-2025gu181dc0e7-2d60-47e4-b28e-d8b740ea1a9f.pdf?sfvrsn=f637cb3b_0",
    ),
    Circular(
        "mcx-circular-491-2025-trading-hours",
        "MCX/TRD/491/2025",
        date(2025, 9, 29),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2025/september/circular-491--2025guc9d9f33c-f9eb-4bd9-a8a2-6a1565fbb79d.pdf?sfvrsn=d4f02716_0",
    ),
    Circular(
        "mcx-circular-636-2025-trading-holidays-2026",
        "MCX/TRD/636/2025",
        date(2025, 12, 15),
        "Trading Holidays for the Calendar Year 2026",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2025/december/circular-636-2025guc877eb08-5a12-46b1-9aee-d233258a4887.pdf?sfvrsn=ee99eafb_0",
    ),
    Circular(
        "mcx-circular-017-2026-holiday-2026-01-15",
        "MCX/TRD/017/2026",
        date(2026, 1, 12),
        "Trading Holiday on account of Municipal Corporation Election in Maharashtra on January 15, 2026",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2026/january/circular-017---2026gu39024.pdf?sfvrsn=fefc2504_0",
    ),
    Circular(
        "mcx-circular-027-2026-live-session-2026-02-01",
        "MCX/TRD/027/2026",
        date(2026, 1, 16),
        "Live Trading Session on February 01, 2026 (Sunday)",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2026/january/circular-027-2026gu37931.pdf?sfvrsn=1ff7cdc1_0",
    ),
    Circular(
        "mcx-circular-068-2026-trading-hours",
        "MCX/TRD/068/2026",
        date(2026, 2, 9),
        "Revision in Trading Hours",
        "https://www.mcxindia.com/docs/default-source/circulars/english/2026/february/circular---068---2026.pdf?sfvrsn=c58e2e11_3",
    ),
)
CIRCULAR_BY_ID = {c.id: c for c in CIRCULARS}


# -- annual holiday tables, as printed ------------------------------------------

#: (date, name, morning, evening) rows copied from each year's circular. Weekend
#: holidays the circular lists separately are included with both sessions
#: closed so the holiday name is retained.
HOLIDAY_CIRCULAR_BY_YEAR = {
    2021: "mcx-circular-931-2020-trading-holidays-2021",
    2022: "mcx-circular-764-2021-trading-holidays-2022",
    2023: "mcx-circular-710-2022-trading-holidays-2023",
    2024: "mcx-circular-872-2023-trading-holidays-2024",
    2025: "mcx-circular-814-2024-trading-holidays-2025",
    2026: "mcx-circular-636-2025-trading-holidays-2026",
}

O, C = "Open", "Closed"
HOLIDAYS: dict[int, tuple[tuple[date, str, str, str], ...]] = {
    2021: (
        (date(2021, 1, 1), "New Year Day", O, C),
        (date(2021, 1, 26), "Republic Day", C, C),
        (date(2021, 3, 11), "Mahashivratri", C, O),
        (date(2021, 3, 29), "Holi (2nd day)", C, O),
        (date(2021, 4, 2), "Good Friday", C, C),
        (date(2021, 4, 14), "Ambedkar Jayanti", C, O),
        (date(2021, 4, 21), "Ram Navmi", C, O),
        (date(2021, 5, 13), "Ramzan ID (Id-Ul-Fitr)", C, O),
        (date(2021, 7, 21), "Bakri Id", C, O),
        (date(2021, 8, 19), "Moharram", C, O),
        (date(2021, 9, 10), "Ganesh Chaturthi", C, O),
        (date(2021, 10, 15), "Dassera", C, O),
        (date(2021, 11, 4), "Diwali - Laxmi Pujan", C, "Muhurat"),
        (date(2021, 11, 5), "Diwali - Balipratipada", C, O),
        (date(2021, 11, 19), "Guru Nanak Jayanti", C, O),
        (date(2021, 4, 25), "Mahavir Jayanti", C, C),
        (date(2021, 5, 1), "Maharashtra Day", C, C),
        (date(2021, 8, 15), "Independence Day", C, C),
        (date(2021, 10, 2), "Gandhi Jayanti", C, C),
        (date(2021, 12, 25), "Christmas", C, C),
    ),
    2022: (
        (date(2022, 1, 26), "Republic Day", C, C),
        (date(2022, 3, 1), "Mahashivratri", C, O),
        (date(2022, 3, 18), "Holi (2nd day)", C, O),
        (date(2022, 4, 14), "Mahavir Jayanti / Ambedkar Jayanti", C, O),
        (date(2022, 4, 15), "Good Friday", C, C),
        (date(2022, 5, 3), "Ramzan Id (Id-Ul-Fitr)", C, O),
        (date(2022, 8, 9), "Moharram", C, O),
        (date(2022, 8, 15), "Independence Day", C, C),
        (date(2022, 8, 31), "Ganesh Chaturthi", C, O),
        (date(2022, 10, 5), "Dassera", C, O),
        (date(2022, 10, 24), "Diwali - Laxmi Pujan", C, "Muhurat"),
        (date(2022, 10, 26), "Diwali Balipratipada", C, O),
        (date(2022, 11, 8), "Gurunanak Jayanti", C, O),
        (date(2022, 1, 1), "New Year Day", C, C),
        (date(2022, 4, 10), "Ram Navami", C, C),
        (date(2022, 5, 1), "Maharashtra Day", C, C),
        (date(2022, 7, 10), "Bakri Id", C, C),
        (date(2022, 10, 2), "Mahatma Gandhi Jayanti", C, C),
        (date(2022, 12, 25), "Christmas", C, C),
    ),
    2023: (
        (date(2023, 1, 26), "Republic Day", C, C),
        (date(2023, 3, 7), "Holi (2nd day)", C, O),
        (date(2023, 3, 30), "Ram Navami", C, O),
        (date(2023, 4, 4), "Mahavir Jayanti", C, O),
        (date(2023, 4, 7), "Good Friday", C, C),
        (date(2023, 4, 14), "Ambedkar Jayanti", C, O),
        (date(2023, 5, 1), "Maharashtra Day", C, O),
        (
            date(2023, 6, 28),
            "Bakri Id",
            C,
            O,
        ),  # moved to 29-Jun-2023 by MCX/TRD/409/2023, see SPECIAL
        (date(2023, 8, 15), "Independence Day", C, C),
        (date(2023, 9, 19), "Ganesh Chaturthi", C, O),
        (date(2023, 10, 2), "Mahatma Gandhi Jayanti", C, C),
        (date(2023, 10, 24), "Dassera", C, O),
        (date(2023, 11, 14), "Diwali Balipratipada", C, O),
        (date(2023, 11, 27), "Guru Nanak Jayanti", C, O),
        (date(2023, 12, 25), "Christmas", C, C),
        (date(2023, 1, 1), "New Year Day", C, C),
        (date(2023, 2, 18), "Mahashivratri", C, C),
        (date(2023, 4, 22), "Ramzan Id (Id-Ul-Fitr)", C, C),
        (date(2023, 7, 29), "Moharram", C, C),
        (date(2023, 11, 12), "Diwali - Laxmi Pujan", C, "Muhurat"),
    ),
    2024: (
        (date(2024, 1, 1), "New Year Day", O, C),
        (date(2024, 1, 26), "Republic Day", C, C),
        (date(2024, 3, 8), "Mahashivratri", C, O),
        (date(2024, 3, 25), "Holi (2nd day)", C, O),
        (date(2024, 3, 29), "Good Friday", C, C),
        (date(2024, 4, 11), "Id-Ul-Fitr (Ramadan Eid)", C, O),
        (date(2024, 4, 17), "Shri Ram Navmi", C, O),
        (date(2024, 5, 1), "Maharashtra Day", C, O),
        (date(2024, 6, 17), "Bakri Id", C, O),
        (date(2024, 7, 17), "Moharram", C, O),
        (date(2024, 8, 15), "Independence Day", C, C),
        (date(2024, 10, 2), "Mahatma Gandhi Jayanti", C, C),
        (date(2024, 11, 1), "Diwali Laxmi Pujan", C, "Muhurat"),
        (date(2024, 11, 15), "Guru Nanak Jayanti", C, O),
        (date(2024, 12, 25), "Christmas", C, C),
        (date(2024, 4, 14), "Dr. Baba Saheb Ambedkar Jayanti", C, C),
        (date(2024, 4, 21), "Shri Mahavir Jayanti", C, C),
        (date(2024, 9, 7), "Ganesh Chaturthi", C, C),
        (date(2024, 10, 12), "Dassera", C, C),
        (date(2024, 11, 2), "Diwali-Balipratipada", C, C),
    ),
    2025: (
        (date(2025, 1, 1), "New Year Day", O, C),
        (date(2025, 2, 26), "Mahashivratri", C, O),
        (date(2025, 3, 14), "Holi (2nd day)", C, O),
        (date(2025, 3, 31), "Id-Ul-Fitr (Ramadan Eid)", C, O),
        (date(2025, 4, 10), "Shri Mahavir Jayanti", C, O),
        (date(2025, 4, 14), "Dr. Baba Saheb Ambedkar Jayanti", C, O),
        (date(2025, 4, 18), "Good Friday", C, C),
        (date(2025, 5, 1), "Maharashtra Day", C, O),
        (date(2025, 8, 15), "Independence Day", C, C),
        (date(2025, 8, 27), "Ganesh Chaturthi", C, O),
        (date(2025, 10, 2), "Mahatma Gandhi Jayanti/ Dassera", C, C),
        (date(2025, 10, 21), "Diwali Laxmi Pujan", C, "Muhurat"),
        (date(2025, 10, 22), "Diwali-Balipratipada", C, O),
        (date(2025, 11, 5), "Guru Nanak Jayanti", C, O),
        (date(2025, 12, 25), "Christmas", C, C),
        (date(2025, 1, 26), "Republic Day", C, C),
        (date(2025, 4, 6), "Shri Ram Navmi", C, C),
        (date(2025, 6, 7), "Bakri Id", C, C),
        (date(2025, 7, 6), "Moharram", C, C),
    ),
    2026: (
        (date(2026, 1, 1), "New Year Day", O, C),
        (date(2026, 1, 26), "Republic Day", C, C),
        (date(2026, 3, 3), "Holi (2nd day)", C, O),
        (date(2026, 3, 26), "Shri Ram Navmi", C, O),
        (date(2026, 3, 31), "Shri Mahavir Jayanti", C, O),
        (date(2026, 4, 3), "Good Friday", C, C),
        (date(2026, 4, 14), "Dr. Baba Saheb Ambedkar Jayanti", C, O),
        (date(2026, 5, 1), "Maharashtra Day", C, O),
        (date(2026, 5, 28), "Bakri Id", C, O),
        (date(2026, 6, 26), "Moharram", C, O),
        (date(2026, 9, 14), "Ganesh Chaturthi", C, O),
        (date(2026, 10, 2), "Mahatma Gandhi Jayanti", C, C),
        (date(2026, 10, 20), "Dassera", C, O),
        (date(2026, 11, 10), "Diwali-Balipratipada", C, O),
        (date(2026, 11, 24), "Guru Nanak Jayanti", C, O),
        (date(2026, 12, 25), "Christmas", C, C),
        (date(2026, 11, 8), "Diwali - Laxmi Pujan", C, "Muhurat"),
    ),
}


@dataclass(frozen=True, slots=True)
class Override:
    """A single-date session stated by a circular other than the annual one."""

    day: date
    state: str
    morning_open: time | None
    morning_close: time | None
    evening_open: time | None
    evening_close: time | None
    special_session: str | None
    holiday_name: str | None
    evidence_id: str
    note: str


#: Muhurat sessions, timed by their own circular. Special (pre-open) sessions
#: are not trading; the trading window is what is stored.
MUHURAT: tuple[Override, ...] = (
    Override(
        date(2021, 11, 4),
        "evening_only",
        None,
        None,
        time(18, 15),
        time(19, 15),
        "muhurat",
        "Diwali - Laxmi Pujan",
        "mcx-circular-671-2021-muhurat-2021-11-04",
        "Muhurat trading 6:15 pm to 7:15 pm IST; no regular session",
    ),
    Override(
        date(2022, 10, 24),
        "evening_only",
        None,
        None,
        time(18, 15),
        time(19, 15),
        "muhurat",
        "Diwali - Laxmi Pujan",
        "mcx-circular-586-2022-muhurat-2022-10-24",
        "Muhurat trading 6:15 pm to 7:15 pm IST; no regular session",
    ),
    Override(
        date(2023, 11, 12),
        "evening_only",
        None,
        None,
        time(18, 15),
        time(19, 15),
        "muhurat",
        "Diwali - Laxmi Pujan",
        "mcx-circular-740-2023-muhurat-2023-11-12",
        "Sunday Muhurat trading 6:15 pm to 7:15 pm IST; no regular session",
    ),
    Override(
        date(2024, 11, 1),
        "evening_only",
        None,
        None,
        time(18, 0),
        time(19, 0),
        "muhurat",
        "Diwali Laxmi Pujan",
        "mcx-circular-704-2024-muhurat-2024-11-01",
        "Muhurat trading 6:00 pm to 7:00 pm IST; no regular session",
    ),
    Override(
        date(2025, 10, 21),
        "morning_only",
        time(13, 45),
        time(14, 45),
        None,
        None,
        "muhurat",
        "Diwali Laxmi Pujan",
        "mcx-circular-477-2025-muhurat-2025-10-21",
        "Muhurat trading 1:45 pm to 2:45 pm IST; no regular session",
    ),
    Override(
        date(2026, 11, 8),
        "evening_only",
        None,
        None,
        None,
        None,
        "muhurat",
        "Diwali - Laxmi Pujan",
        "mcx-circular-636-2025-trading-holidays-2026",
        "Sunday Muhurat trading announced; 'Timings of Muhurat trading shall be notified subsequently' and no timing circular had been issued as of 2026-09-15. The feed lists this date under MCX_UNVERIFIED_DATES",
    ),
)

#: Other single-date sessions: election holidays, budget-day live sessions,
#: and the two days whose open was delayed.
SPECIAL: tuple[Override, ...] = (
    Override(
        date(2023, 6, 28),
        "open",
        MORNING_OPEN,
        MORNING_CLOSE,
        EVENING_OPEN,
        None,
        None,
        None,
        "mcx-circular-409-2023-holiday-revision-bakri-id",
        "Bakri Id holiday moved from 28-Jun-2023 to 29-Jun-2023 by MCX/TRD/409/2023; 28-Jun-2023 is a full trading day",
    ),
    Override(
        date(2023, 6, 29),
        "evening_only",
        None,
        None,
        EVENING_OPEN,
        None,
        None,
        "Bakri Id",
        "mcx-circular-409-2023-holiday-revision-bakri-id",
        "Bakri Id holiday moved here from 28-Jun-2023 by MCX/TRD/409/2023; morning closed, evening 5:00 PM to 11:30 PM",
    ),
    Override(
        date(2023, 10, 16),
        "open",
        time(10, 45),
        MORNING_CLOSE,
        EVENING_OPEN,
        None,
        "delayed_open",
        None,
        "mcx-circular-700-2023-trading-hours-2023-10-16",
        "Trade start 10:45 am (special session 10:30-10:44) for the new trading platform go-live; evening close as notified",
    ),
    Override(
        date(2024, 1, 22),
        "evening_only",
        None,
        None,
        EVENING_OPEN,
        CLOSE_STANDARD,
        None,
        "Holiday declared by Government of Maharashtra",
        "mcx-circular-037-2024-holiday-2024-01-22",
        "Morning session closed; trading 05:00 pm to 11:55 pm",
    ),
    Override(
        date(2024, 2, 13),
        "open",
        time(13, 0),
        MORNING_CLOSE,
        EVENING_OPEN,
        None,
        "delayed_open",
        None,
        "mcx-circular-089-2024-trading-hours-2024-02-13",
        "Market opened 01:00 pm after three successive revisions (MCX/CSQ/087, 088, 089/2024) 'due to delay in file generation'",
    ),
    Override(
        date(2024, 5, 20),
        "evening_only",
        None,
        None,
        EVENING_OPEN,
        CLOSE_DST,
        None,
        "General Parliamentary Elections 2024",
        "mcx-circular-219-2024-holiday-2024-05-20",
        "Morning session closed; evening 5:00 pm to 11:30 pm",
    ),
    Override(
        date(2024, 11, 20),
        "evening_only",
        None,
        None,
        EVENING_OPEN,
        CLOSE_STANDARD,
        None,
        "Maharashtra Legislative Assembly General Election",
        "mcx-circular-738-2024-holiday-2024-11-20",
        "Morning session closed; evening 5:00 pm to 11:55 pm",
    ),
    Override(
        date(2025, 2, 1),
        "morning_only",
        MORNING_OPEN,
        MORNING_CLOSE,
        None,
        None,
        "budget_live_session",
        None,
        "mcx-circular-841-2024-live-session-2025-02-01",
        "Saturday special live trading session 09:00 am to 05:00 pm for the Union Budget",
    ),
    Override(
        date(2026, 1, 15),
        "evening_only",
        None,
        None,
        EVENING_OPEN,
        CLOSE_STANDARD,
        None,
        "Maharashtra Municipal Corporation Election",
        "mcx-circular-017-2026-holiday-2026-01-15",
        "Morning session closed; evening 5:00 pm to 11:55 pm. Not encoded in feed mcx_calendar.py as of 2026-09-15",
    ),
    Override(
        date(2026, 2, 1),
        "morning_only",
        MORNING_OPEN,
        MORNING_CLOSE,
        None,
        None,
        "budget_live_session",
        None,
        "mcx-circular-027-2026-live-session-2026-02-01",
        "Sunday special live trading session 09:00 am to 05:00 pm for the Union Budget. Not encoded in feed mcx_calendar.py as of 2026-09-15",
    ),
)


@dataclass(frozen=True, slots=True)
class EveningClosePeriod:
    start: date
    end: date
    close: time
    evidence_id: str


#: Retained *Revision in Trading Hours* circulars, in force order. A DST-start
#: circular names no end date; it is bounded by the next DST-end circular, and
#: the last one by the date the feed verified coverage through.
EVENING_CLOSE_PERIODS: tuple[EveningClosePeriod, ...] = (
    EveningClosePeriod(
        date(2020, 11, 2), date(2021, 3, 12), CLOSE_STANDARD, "mcx-circular-763-2020-trading-hours"
    ),
    EveningClosePeriod(
        date(2021, 3, 15), date(2021, 11, 7), CLOSE_DST, "mcx-circular-085-2021-trading-hours"
    ),
    EveningClosePeriod(
        date(2021, 11, 8), date(2022, 3, 11), CLOSE_STANDARD, "mcx-circular-646-2021-trading-hours"
    ),
    EveningClosePeriod(
        date(2022, 3, 14), date(2022, 11, 6), CLOSE_DST, "mcx-circular-076-2022-trading-hours"
    ),
    EveningClosePeriod(
        date(2022, 11, 7), date(2023, 3, 10), CLOSE_STANDARD, "mcx-circular-574-2022-trading-hours"
    ),
    EveningClosePeriod(
        date(2023, 3, 13), date(2023, 11, 5), CLOSE_DST, "mcx-circular-095-2023-trading-hours"
    ),
    EveningClosePeriod(
        date(2023, 11, 6), date(2024, 3, 8), CLOSE_STANDARD, "mcx-circular-675-2023-trading-hours"
    ),
    EveningClosePeriod(
        date(2024, 3, 11), date(2024, 11, 1), CLOSE_DST, "mcx-circular-082-2024-trading-hours"
    ),
    EveningClosePeriod(
        date(2024, 11, 4), date(2025, 3, 7), CLOSE_STANDARD, "mcx-circular-643-2024-trading-hours"
    ),
    EveningClosePeriod(
        date(2025, 3, 10), date(2025, 11, 2), CLOSE_DST, "mcx-circular-071-2025-trading-hours"
    ),
    EveningClosePeriod(
        date(2025, 11, 3), date(2026, 3, 6), CLOSE_STANDARD, "mcx-circular-491-2025-trading-hours"
    ),
    EveningClosePeriod(
        date(2026, 3, 9), date(2026, 10, 31), CLOSE_DST, "mcx-circular-068-2026-trading-hours"
    ),
)


def evening_close(day: date) -> tuple[time, str | None]:
    """(close, evidence id) for the date; (fallback, None) past the retained circulars."""
    for period in EVENING_CLOSE_PERIODS:
        if period.start <= day <= period.end:
            return period.close, period.evidence_id
    return FALLBACK_CLOSE, None


# -- builder ----------------------------------------------------------------------

STATES = ("open", "morning_only", "evening_only", "closed")

SESSIONS_SCHEMA = pa.schema(
    [
        ("session_date", pa.date32()),
        ("weekday", pa.string()),
        ("state", pa.string()),
        ("morning_open_ist", pa.time64("us")),
        ("morning_close_ist", pa.time64("us")),
        ("evening_open_ist", pa.time64("us")),
        ("evening_close_ist", pa.time64("us")),
        ("hours_notified", pa.bool_()),
        ("special_session", pa.string()),
        ("holiday_name", pa.string()),
        ("evidence_ids", pa.list_(pa.string())),
        ("note", pa.string()),
        ("available_utc", pa.timestamp("us", tz="UTC")),
        ("clock_quality", pa.string()),
    ]
)

SESSIONS_COLUMN_MEANINGS = {
    "session_date": "Asia/Kolkata civil date",
    "weekday": "three-letter weekday name",
    "state": "open | morning_only | evening_only | closed",
    "morning_open_ist": "start of the first trading window, IST; null when the morning session is shut",
    "morning_close_ist": "end of the first trading window, IST",
    "evening_open_ist": "start of the second trading window, IST; null when the evening session is shut",
    "evening_close_ist": "end of the second trading window, IST: 23:30 while New York is on daylight time, 23:55 otherwise, per the Revision in Trading Hours circular in force; Muhurat/special windows carry their own times",
    "hours_notified": "true when a retained circular states this date's evening close; false for the 23:30 fallback past the last retained circular and for untimed announced sessions",
    "special_session": "muhurat | budget_live_session | delayed_open | null",
    "holiday_name": "holiday as printed in the circular, null on ordinary days",
    "evidence_ids": "ids of the captured circulars that state something about this date (build/mcx/raw/circulars/<id>.pdf)",
    "note": "free text for anything non-standard",
    "available_utc": "00:00 UTC of the day after the latest issue date among this row's evidence_ids: the instant by which every circular the row relies on had been published",
    "clock_quality": "published when every stated time is backed by a circular in evidence_ids; unvintaged when the row asserts the 23:30 fallback evening close past the last retained Revision in Trading Hours circular",
}


def circular_available_utc(evidence_ids: list[str]) -> datetime:
    """00:00 UTC of the day after the latest issue date among the circulars."""
    latest = max(CIRCULAR_BY_ID[evidence_id].issued for evidence_id in evidence_ids)
    return datetime.combine(latest + timedelta(days=1), time(0, 0), tzinfo=UTC)


def session_close_utc(row: dict) -> datetime | None:
    """The instant the day's last trading window closes, in UTC; None on closed days.

    Where the evening close is the fallback rather than a notified one, the
    later of the two published closes (23:55 IST) is used, so the instant is
    never earlier than the true close.
    """
    if row["state"] == "closed":
        return None
    if row["evening_open_ist"] is not None:
        close = row["evening_close_ist"]
        if close is None:
            return None
        if not row["hours_notified"] and row["special_session"] is None:
            close = max(CLOSE_DST, CLOSE_STANDARD)
    else:
        close = row["morning_close_ist"]
    if close is None:
        return None
    return datetime.combine(row["session_date"], close, tzinfo=IST).astimezone(UTC)


def _holiday_rows(year: int) -> dict[date, tuple[str, str, str]]:
    return {day: (name, morning, evening) for day, name, morning, evening in HOLIDAYS[year]}


def build_rows(first: date = FIRST_DATE, last: date = LAST_DATE) -> list[dict]:
    overrides = {o.day: o for o in (*MUHURAT, *SPECIAL)}
    rows: list[dict] = []
    day = first
    while day <= last:
        holidays = _holiday_rows(day.year)
        annual = HOLIDAY_CIRCULAR_BY_YEAR[day.year]
        close, close_evidence = evening_close(day)
        evidence = [annual]
        if close_evidence:
            evidence.append(close_evidence)
        weekend = day.weekday() >= 5
        holiday_name = holidays[day][0] if day in holidays else None
        override = overrides.get(day)
        row = {
            "session_date": day,
            "weekday": calendar.day_abbr[day.weekday()],
            "state": "closed",
            "morning_open_ist": None,
            "morning_close_ist": None,
            "evening_open_ist": None,
            "evening_close_ist": None,
            "hours_notified": close_evidence is not None,
            "special_session": None,
            "holiday_name": holiday_name,
            "evidence_ids": evidence,
            "note": None,
        }
        if override is not None:
            row.update(
                state=override.state,
                morning_open_ist=override.morning_open,
                morning_close_ist=override.morning_close,
                evening_open_ist=override.evening_open,
                evening_close_ist=(
                    close
                    if override.evening_open and override.evening_close is None
                    else override.evening_close
                ),
                special_session=override.special_session,
                holiday_name=(
                    override.holiday_name
                    if override.state == "open"
                    else override.holiday_name or holiday_name
                ),
                note=override.note,
            )
            if (
                override.special_session == "muhurat"
                and override.evening_open is None
                and override.morning_open is None
            ):
                row["hours_notified"] = False
            if override.day == date(2024, 2, 13):
                row["evidence_ids"].extend(
                    [
                        "mcx-circular-087-2024-trading-hours-2024-02-13",
                        "mcx-circular-088-2024-trading-hours-2024-02-13",
                    ]
                )
            if override.evidence_id not in row["evidence_ids"]:
                row["evidence_ids"].append(override.evidence_id)
        elif weekend:
            row["state"] = "closed"
        elif day in holidays:
            _, morning, evening = holidays[day]
            if morning == O and evening == C:
                row.update(
                    state="morning_only",
                    morning_open_ist=MORNING_OPEN,
                    morning_close_ist=MORNING_CLOSE,
                )
            elif morning == C and evening == O:
                row.update(
                    state="evening_only", evening_open_ist=EVENING_OPEN, evening_close_ist=close
                )
            elif morning == C and evening == C:
                row["state"] = "closed"
            else:
                raise ValueError(f"unhandled holiday states {morning}/{evening} on {day}")
        else:
            row.update(
                state="open",
                morning_open_ist=MORNING_OPEN,
                morning_close_ist=MORNING_CLOSE,
                evening_open_ist=EVENING_OPEN,
                evening_close_ist=close,
            )
        if row["state"] not in STATES:
            raise ValueError(f"bad state {row['state']} on {day}")
        row["available_utc"] = circular_available_utc(row["evidence_ids"])
        fallback_close = (
            row["evening_open_ist"] is not None
            and row["evening_close_ist"] is not None
            and not row["hours_notified"]
            and row["special_session"] is None
        )
        row["clock_quality"] = "unvintaged" if fallback_close else "published"
        rows.append(row)
        day += timedelta(days=1)
    return rows


def to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=SESSIONS_SCHEMA)


def capture_circulars(client: McxClient, lane_root: Path | None = None) -> dict[str, str]:
    """Download every catalogued circular that is not yet on disk; returns id -> sha256."""
    root = lane_root or common.lane_root(LANE)
    from .bhavcopy import read_index

    index = read_index(root)
    digests: dict[str, str] = {}
    for circular in CIRCULARS:
        path = root / "raw" / "circulars" / f"{circular.id}.pdf"
        entry = index.get(str(path))
        if entry and path.exists() and common.sha256_bytes(path.read_bytes()) == entry["sha256"]:
            digests[circular.id] = entry["sha256"]
            continue
        capture = client.capture(
            circular.url,
            f"circulars/{circular.id}.pdf",
            headers={
                "Accept": "application/pdf,*/*",
                "Referer": "https://www.mcxindia.com/circulars/all-circulars",
            },
        )
        if not capture.body.startswith(b"%PDF-"):
            raise RuntimeError(f"{circular.id}: response is not a PDF ({capture.body[:40]!r})")
        digests[circular.id] = capture.sha256
    return digests


def write_sessions(*, command: str, lane_root: Path | None = None) -> Path:
    import pyarrow.parquet as pq

    from .bhavcopy import read_index

    root = lane_root or common.lane_root(LANE)
    rows = build_rows()
    table = to_table(rows)
    out = root / "mcx_sessions.parquet"
    pq.write_table(table, out, compression="zstd")
    index = read_index(root)
    sources = []
    for circular in CIRCULARS:
        entry = index.get(str(root / "raw" / "circulars" / f"{circular.id}.pdf"))
        if entry is None:
            raise RuntimeError(f"{circular.id} has no capture; run capture_circulars first")
        sources.append(
            {
                "id": circular.id,
                "circular": circular.number,
                "issued": circular.issued.isoformat(),
                "title": circular.title,
                "url": entry["url"],
                "fetched_at_utc": entry["fetched_at_utc"],
                "sha256": entry["sha256"],
                "bytes": int(entry["bytes"]),
                "local_path": common.sidecar_path(entry["path"]),
            }
        )
    states = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    quality_counts = {}
    for row in rows:
        quality_counts[row["clock_quality"]] = quality_counts.get(row["clock_quality"], 0) + 1
    meta = {
        "name": "mcx_sessions",
        "rows": table.num_rows,
        "columns": [
            {"name": f.name, "type": str(f.type), "meaning": SESSIONS_COLUMN_MEANINGS[f.name]}
            for f in SESSIONS_SCHEMA
        ],
        "date_range": {
            "start": FIRST_DATE.isoformat(),
            "end": LAST_DATE.isoformat(),
            "states": states,
        },
        "sources": sources,
        "clock_quality": {
            "file": "mixed" if len(quality_counts) > 1 else next(iter(quality_counts)),
            "row_column": "clock_quality",
            "rows": quality_counts,
            "clocks": {"published": "available_utc"},
            "rule": (
                "Each row's state and hours are stated by the circulars in evidence_ids; available_utc is "
                "00:00 UTC of the day after the latest of their issue dates (the annual trading-holiday "
                "circular alone for ordinary and holiday dates it governs, plus the Revision in Trading Hours "
                "circular that states the evening close, plus any single-date circular such as an election "
                "closure, a Muhurat timing or a budget-day live session). Rows classed unvintaged assert the "
                "23:30 fallback evening close past the last retained hours circular: their state is published "
                "at available_utc but their close time has no circular behind it."
            ),
        },
        "build": {**common.git_identity(Path(__file__).resolve().parent), "command": command},
        "notes": [
            "Hours are for internationally referenceable non-agricultural commodities (gold, silver, copper, crude oil); agricultural contracts close earlier.",
            "Weekend dates are closed unless a circular states a session (budget-day live sessions, Sunday Muhurat trading).",
            "Muhurat and budget-day sessions are stored with their own trading windows in the open/close columns and flagged in special_session; the state column keeps the four-value enum.",
            "Evening close past 2026-10-31 is the 23:30 fallback with hours_notified=false, mirroring feed mcx_calendar.py; the US DST-end 2026 circular had not been issued as of 2026-09-15.",
            "2026 differs from feed mcx_calendar.py on two circular-backed dates: 2026-01-15 (evening only, MCX/TRD/017/2026) and Sunday 2026-02-01 (special live session, MCX/TRD/027/2026).",
        ],
    }
    common.write_meta(out, meta)
    (root / "mcx_sessions.evidence.json").write_text(json.dumps(sources, indent=2) + "\n")
    return out
