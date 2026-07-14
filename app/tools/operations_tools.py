"""
Simulated backend tools the Operations Agent calls.

These stand in for a real MCP server / booking backend. They're written as
plain functions with typed signatures and structured dict returns -- the
same shape a real MCP tool_result or REST API response would have -- so
swapping in a real server later means changing only the call site inside
OperationsAgent, not the calling convention.

check_table_availability is deterministic (hashed on date+time+branch)
rather than random, so repeated calls with the same inputs are reproducible
during grading/evaluation. Date and time are normalized to a canonical form
BEFORE hashing, so "8pm", "8:00 PM", and "20:00" -- which all describe the
same real-world time -- hash to the same seed and return the same result.
Without this, the same slot could report different availability depending
on how the LLM happened to format its tool call arguments.
"""
from datetime import datetime
from typing import Dict, Any, List
import hashlib
import re

BRANCHES = ["Downtown", "Maadi", "New Cairo"]

_TODAY_SPECIALS = {
    "Downtown": "Grilled Seabass with Lemon Butter",
    "Maadi": "Chicken Fattah",
    "New Cairo": "Truffle Mushroom Risotto",
}

_LOYALTY_POINTS = {
    "user_001": 620,
    "user_002": 150,
    "user_003": 1240,
}

_bookings: List[Dict[str, Any]] = []

_TIME_PATTERN = re.compile(r'^\s*(\d{1,2})(?::(\d{2}))?\s*([APap][Mm])?\s*$')
_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]


def normalize_time(raw_time: str) -> str:
    """Normalize '8pm' / '8:00 PM' / '20:00' etc. into canonical 24h 'HH:MM'.
    Falls back to the trimmed input unchanged if it doesn't match a
    recognized pattern (still deterministic, just not normalized)."""
    match = _TIME_PATTERN.match(raw_time or "")
    if not match:
        return (raw_time or "").strip()

    hour_str, minute_str, meridiem = match.groups()
    hour = int(hour_str)
    minute = int(minute_str) if minute_str else 0

    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return (raw_time or "").strip()

    return f"{hour:02d}:{minute:02d}"


def normalize_date(raw_date: str) -> str:
    """Normalize common date formats into canonical 'YYYY-MM-DD'.
    Falls back to the trimmed input unchanged if it doesn't match any
    recognized format."""
    raw_date = (raw_date or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(raw_date, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw_date


def _deterministic_slot_count(seed: str, mod: int = 6) -> int:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return int(digest, 16) % mod


def check_table_availability(date: str, time: str, branch: str) -> Dict[str, Any]:
    if branch not in BRANCHES:
        return {"success": False, "error": f"Unknown branch '{branch}'. Valid branches: {BRANCHES}"}

    norm_date = normalize_date(date)
    norm_time = normalize_time(time)

    available_tables = _deterministic_slot_count(f"{norm_date}|{norm_time}|{branch}")
    return {
        "success": True,
        "branch": branch,
        "date": norm_date,
        "time": norm_time,
        "available_tables": available_tables,
        "status": "available" if available_tables > 0 else "fully_booked",
    }


def book_table(name: str, date: str, time: str, branch: str) -> Dict[str, Any]:
    availability = check_table_availability(date, time, branch)
    if not availability["success"]:
        return availability
    if availability["status"] == "fully_booked":
        return {
            "success": False,
            "message": f"No tables available at {branch} on {availability['date']} at {availability['time']}.",
        }

    booking_id = f"BK{len(_bookings) + 1:04d}"
    booking = {
        "booking_id": booking_id,
        "name": name,
        "date": availability["date"],
        "time": availability["time"],
        "branch": branch,
    }
    _bookings.append(booking)
    return {"success": True, "booking": booking}


def get_today_special(branch: str) -> Dict[str, Any]:
    if branch not in BRANCHES:
        return {"success": False, "error": f"Unknown branch '{branch}'. Valid branches: {BRANCHES}"}
    return {
        "success": True,
        "branch": branch,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "special": _TODAY_SPECIALS[branch],
    }


def check_loyalty_points(user_id: str) -> Dict[str, Any]:
    if user_id not in _LOYALTY_POINTS:
        return {"success": False, "error": f"No loyalty record found for user_id '{user_id}'."}
    points = _LOYALTY_POINTS[user_id]
    return {
        "success": True,
        "user_id": user_id,
        "points": points,
        "redeemable": points >= 500,
        "discount_available_egp": 100 if points >= 500 else 0,
    }
