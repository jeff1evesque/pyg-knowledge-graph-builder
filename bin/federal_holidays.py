#!/usr/bin/env python3
"""Say which US federal holiday a day is, if it is one.

    bin/federal_holidays.py YYYY-MM-DD

Prints the day's holiday, such as "Columbus Day", or nothing, and exits 0. Exits 2
when the argument is not a date written YYYY-MM-DD.

bin/daily_run.sh asks it about the data day. A source in
PYG_HOLIDAY_OPTIONAL_SOURCE_PATHS, such as the SEC filings day file, is published only
on federal workdays: EDGAR takes no filings on a federal holiday, and upstream writes no
file for a day without filings. The market can still be open that day, as on Columbus
Day and Veterans Day, so on a holiday the day builds without that file rather than
being skipped. On any other day a missing file means upstream is late.

The holidays are the eleven of 5 U.S.C. 6103(a), Juneteenth from 2021, the year it
became one. One that falls on a Saturday is observed on the Friday before, and one on a
Sunday on the Monday after, which is when federal offices close (5 U.S.C. 6103(b),
Executive Order 11582). The observed day is the holiday here, and the weekend day itself
is not: offices are closed on it anyway. So a Saturday New Year's Day makes Friday Dec 31
of the year before a holiday, and the market trades that day.

Not here: Inauguration Day, a holiday only around Washington (6103(c)), and one-off
closings by executive order. bin/daily_run.sh takes those in PYG_EXTRA_CLOSED_DAYS.
"""

import calendar
import datetime
import re
import sys

MONDAY, THURSDAY, SATURDAY, SUNDAY = 0, 3, 5, 6

# Month, day and name of the holidays on a fixed date.
FIXED = [
    (1, 1, "New Year's Day"),
    (6, 19, "Juneteenth National Independence Day"),
    (7, 4, "Independence Day"),
    (11, 11, "Veterans Day"),
    (12, 25, "Christmas Day"),
]
JUNETEENTH_FROM = 2021

# Month, weekday, which one (1 the first, -1 the last) and name of the holidays on a
# weekday of a month.
FLOATING = [
    (1, MONDAY, 3, "Birthday of Martin Luther King, Jr."),
    (2, MONDAY, 3, "Washington's Birthday"),
    (5, MONDAY, -1, "Memorial Day"),
    (9, MONDAY, 1, "Labor Day"),
    (10, MONDAY, 2, "Columbus Day"),
    (11, THURSDAY, 4, "Thanksgiving Day"),
]


def nth_weekday(year, month, weekday, n):
    """The nth `weekday` of the month, counting from 1; -1 is the last."""
    if n < 0:
        last = datetime.date(year, month, calendar.monthrange(year, month)[1])
        return last - datetime.timedelta(days=(last.weekday() - weekday) % 7)
    first = datetime.date(year, month, 1)
    return first + datetime.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def observed(day):
    """The day federal offices close for a holiday that falls on `day`."""
    if day.weekday() == SATURDAY:
        return day - datetime.timedelta(days=1)
    if day.weekday() == SUNDAY:
        return day + datetime.timedelta(days=1)
    return day


def holidays(year):
    """{day offices close: name} for the year's holidays. A Saturday New Year's Day is
    closed for on Dec 31, so that day is among the next year's holidays."""
    out = {}
    for month, day, name in FIXED:
        if month == 6 and year < JUNETEENTH_FROM:
            continue
        actual = datetime.date(year, month, day)
        closed = observed(actual)
        out[closed] = name if closed == actual else f"{name} (observed)"
    for month, weekday, n, name in FLOATING:
        out[nth_weekday(year, month, weekday, n)] = name
    return out


def holiday(day):
    """The federal holiday `day` is, or None."""
    return holidays(day.year).get(day) or holidays(day.year + 1).get(day)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: federal_holidays.py YYYY-MM-DD", file=sys.stderr)
        return 2
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args[0]):
            raise ValueError(args[0])
        day = datetime.date.fromisoformat(args[0])
    except ValueError:
        print(f"not a date written YYYY-MM-DD: {args[0]!r}", file=sys.stderr)
        return 2
    name = holiday(day)
    if name:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
