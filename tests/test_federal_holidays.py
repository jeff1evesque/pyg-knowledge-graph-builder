"""Pin the calendar in `bin/federal_holidays.py`.

WHY THIS IS WORTH A TEST
------------------------
bin/daily_run.sh builds a day without its holiday-optional sources, such as the SEC
filings day file, only when this calendar says the day is a federal holiday. A wrong
"yes" on a workday publishes a graph missing that day's filings, and a publish can't
be taken back. A wrong "no" on a holiday skips a day the market traded, the way
2026-10-12, Columbus Day, was skipped before the calendar existed.

The rules are fixed dates and weekday counts, so each is pinned where it can go wrong:
a month with five of the weekday, a holiday on a weekend, and a Saturday New Year's
Day observed in the year before.
"""

import importlib.util
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "bin" / "federal_holidays.py"


def _load_tool():
    """Import by path -- bin/ is scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("federal_holidays_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.mark.parametrize("day, name", [
    (date(2026, 10, 12), "Columbus Day"),
    (date(2026, 11, 11), "Veterans Day"),
], ids=["columbus-day", "veterans-day"])
def test_the_days_the_market_trades_and_the_sec_is_closed(tool, day, name):
    assert tool.holiday(day) == name


def test_a_saturday_new_years_day_is_observed_on_the_friday_before(tool):
    """2028-01-01 is a Saturday, so offices close on Friday 2027-12-31, a day the
    market trades."""
    assert tool.holiday(date(2027, 12, 31)) == "New Year's Day (observed)"
    assert tool.holiday(date(2028, 1, 1)) is None


@pytest.mark.parametrize("day", [
    date(2026, 10, 13),
    date(2026, 10, 10),
    date(2026, 7, 4),
], ids=["the-day-after-columbus-day", "a-saturday", "a-saturday-holiday"])
def test_not_holidays(tool, day):
    """A Saturday holiday closes offices on the Friday before; the Saturday is closed
    anyway."""
    assert tool.holiday(day) is None


def test_a_saturday_holiday_moves_to_the_friday_before(tool):
    assert tool.holiday(date(2026, 7, 3)) == "Independence Day (observed)"


def test_a_sunday_holiday_moves_to_the_monday_after(tool):
    """2027-07-04 is a Sunday."""
    assert tool.holiday(date(2027, 7, 4)) is None
    assert tool.holiday(date(2027, 7, 5)) == "Independence Day (observed)"


def test_thanksgiving_is_the_fourth_thursday_not_the_last(tool):
    """November 2029 has five Thursdays: 1, 8, 15, 22 and 29."""
    assert tool.holiday(date(2029, 11, 22)) == "Thanksgiving Day"
    assert tool.holiday(date(2029, 11, 29)) is None


def test_memorial_day_is_the_last_monday_of_may(tool):
    """May 2028 has five Mondays: 1, 8, 15, 22 and 29."""
    assert tool.holiday(date(2028, 5, 29)) == "Memorial Day"
    assert tool.holiday(date(2028, 5, 22)) is None


def test_juneteenth_counts_from_2021(tool):
    """2021-06-19 was a Saturday, so offices closed on Friday the 18th."""
    assert tool.holiday(date(2021, 6, 18)) == "Juneteenth National Independence Day (observed)"
    assert tool.holiday(date(2020, 6, 19)) is None


def test_no_two_holidays_share_a_day(tool):
    """Eleven holidays a year, each on a day of its own."""
    for year in range(2021, 2041):
        assert len(tool.holidays(year)) == 11, year


def test_the_command_prints_the_holiday_or_nothing():
    """bin/daily_run.sh reads the name from stdout and refuses on a non-zero exit."""
    def run(arg):
        return subprocess.run([sys.executable, str(TOOL), arg], capture_output=True, text=True)

    r = run("2026-10-12")
    assert (r.returncode, r.stdout) == (0, "Columbus Day\n")
    r = run("2026-10-13")
    assert (r.returncode, r.stdout) == (0, "")
    for not_a_day in ("2026-10-32", "20261012", "Oct 12"):
        assert run(not_a_day).returncode == 2, not_a_day
