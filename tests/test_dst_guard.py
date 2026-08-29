"""
Tests for scripts/dst_guard.sh — the New York time-zone guard behind
refresh_data.yml's daily build (Full-Universe spec point 18/R).

Resilience fix (2026-08-28): a real incident saw scheduled GitHub Actions
firings delayed by 11+ hours, landing outside the old exact "hour == 16"
match on every subsequent firing for days -- no fresh data got committed
at all in the meantime, even though every run reported "success" (skipped
steps aren't a failure).

Follow-up fix (same day): the same-day dedup introduced by the resilience
fix was itself buggy -- a manual workflow_dispatch run BEFORE market close
(observed: 2026-08-28, triggered ~10:57 EDT, hours before the 16:00 NY
close) satisfied "already refreshed today" purely by calendar date, which
then suppressed the REAL after-close refresh for that same day for the
rest of the day. The dedup now also requires the earlier same-day update
to itself have landed at or after 16:00 NY before it counts.

The guard now decides, in order: (1) workflow_dispatch always runs; (2)
already refreshed today (NY calendar date) AT OR AFTER 16:00 NY -> skip --
an earlier INTRADAY same-day update does not count; (3) more than one full
weekday behind (skipping weekends) -> run regardless of hour, to catch up;
(4) local NY hour >= 16 -> run; (5) otherwise skip.

Run with: pytest tests/ -v
"""

import subprocess
from pathlib import Path

SCRIPT = str(Path(__file__).parent.parent / "scripts" / "dst_guard.sh")


def run_guard(event_name, hour=None, today=None, last_updated=None, last_updated_hour=None):
    args = ["bash", SCRIPT, event_name]
    if hour is not None:
        args.append(hour)
        if today is not None:
            args.append(today)
            if last_updated is not None:
                args.append(last_updated)
                if last_updated_hour is not None:
                    args.append(last_updated_hour)
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_workflow_dispatch_always_runs_regardless_of_hour():
    for hour in ("00", "09", "15", "16", "17", "23"):
        assert run_guard("workflow_dispatch", hour) == "true"


def test_workflow_dispatch_ignores_hour_entirely_even_without_override():
    # No override -> would call the real `date` command, but workflow_dispatch
    # must short-circuit before ever reaching that branch.
    assert run_guard("workflow_dispatch") == "true"


# ── Same-day dedup: already refreshed today AFTER CLOSE -> skip ──

def test_skips_when_already_refreshed_after_close_today_regardless_of_asking_hour():
    for hour in ("00", "16", "17", "23"):
        assert run_guard("schedule", hour, "2026-08-28", "2026-08-28", "16") == "false"


def test_a_delayed_firing_landing_after_close_also_counts_as_the_real_refresh():
    # The last update itself was delayed into hour 22 (not exactly 16) --
    # still a real after-close refresh, still dedups later firings the same day.
    assert run_guard("schedule", "23", "2026-08-28", "2026-08-28", "22") == "false"


# ── Follow-up fix: an INTRADAY same-day update (e.g. a manual dispatch
# before close) must NOT block the real after-close refresh ──

def test_intraday_manual_update_today_does_not_block_the_real_after_close_refresh():
    # Exact incident reproduction: a manual run landed at NY hour 11 (long
    # before the 16:00 close). A LATER firing the same day, even delayed to
    # hour 22, must still run -- otherwise that day's real closing data is
    # never captured at all.
    assert run_guard("schedule", "22", "2026-08-28", "2026-08-28", "11") == "true"


def test_after_the_real_post_close_refresh_runs_a_further_same_day_firing_is_deduped():
    # Continuation of the scenario above: once THAT hour-22 run has landed
    # (updating last_updated_hour to 22), a subsequent same-day firing must
    # go back to being deduped normally.
    assert run_guard("schedule", "23", "2026-08-28", "2026-08-28", "22") == "false"


def test_intraday_update_today_still_gates_normally_on_the_hour_window():
    # Before 16:00 NY, an intraday same-day update changes nothing -- still
    # too early, same as if no update had happened today at all.
    assert run_guard("schedule", "14", "2026-08-28", "2026-08-28", "11") == "false"


def test_missing_last_updated_hour_defaults_to_intraday_not_post_close():
    # When only a date override is given (no hour), the same-day match must
    # NOT be treated as an after-close refresh by default -- this is the
    # safe default that the bug fix depends on.
    assert run_guard("schedule", "20", "2026-08-28", "2026-08-28") == "true"


# ── Normal path: not stale, gated by the (now widened) hour window ──

def test_runs_at_16_local_ny_time_when_not_yet_refreshed_today():
    assert run_guard("schedule", "16", "2026-08-28", "2026-08-27", "17") == "true"


def test_runs_when_delayed_into_a_later_hour_the_same_day():
    # The core resilience fix: a firing delayed a few hours within the SAME
    # day (e.g. hour 17-23) must still run instead of being silently
    # skipped forever, as long as today hasn't been refreshed yet.
    for hour in ("17", "18", "20", "23"):
        assert run_guard("schedule", hour, "2026-08-28", "2026-08-27", "17") == "true"


def test_skips_before_16_when_not_stale_enough_to_catch_up():
    for hour in ("00", "09", "14", "15"):
        assert run_guard("schedule", hour, "2026-08-28", "2026-08-27", "17") == "false"


def test_octal_danger_hours_08_and_09_do_not_crash_the_numeric_comparison():
    # `date +%H` zero-pads single-digit hours ("08", "09"); a naive `[ -ge ]`
    # comparison misinterprets a leading zero as octal, and 8/9 are invalid
    # octal digits -- this would error out, not just misclassify. Also
    # covers the SAME zero-padding risk for last_updated_hour.
    assert run_guard("schedule", "08", "2026-08-28", "2026-08-27", "17") == "false"
    assert run_guard("schedule", "09", "2026-08-28", "2026-08-27", "17") == "false"
    assert run_guard("schedule", "20", "2026-08-28", "2026-08-28", "08") == "true"  # 08 wasn't post-close


# ── Weekend-aware staleness: a normal Monday must NOT look catch-up-stale ──

def test_monday_with_friday_as_last_update_is_not_treated_as_stale():
    # 2026-08-31 is a Monday; Friday 2026-08-28 is the expected last refresh
    # under completely normal weekday-only scheduling -- must behave exactly
    # like any other single-day gap (gated by hour, not an automatic catch-up).
    assert run_guard("schedule", "15", "2026-08-31", "2026-08-28", "17") == "false"
    assert run_guard("schedule", "16", "2026-08-31", "2026-08-28", "17") == "true"


def test_saturday_or_sunday_with_friday_as_last_update_is_not_stale():
    # Cron only fires Mon-Fri, but a heavily-delayed Friday firing could
    # physically execute over the weekend -- must still use Friday as the
    # weekend-aware baseline, not treat a two-calendar-day gap as catch-up.
    assert run_guard("schedule", "10", "2026-09-05", "2026-09-04", "17") == "false"  # Saturday
    assert run_guard("schedule", "10", "2026-09-06", "2026-09-04", "17") == "false"  # Sunday


# ── Catch-up: more than one full weekday behind -> run regardless of hour ──

def test_catches_up_when_more_than_one_weekday_behind_even_at_an_early_hour():
    # Last update 2026-08-24 (Monday), "today" 2026-08-26 (Wednesday) means
    # the entire Tuesday refresh was missed -- this is exactly the incident
    # scenario (delayed firing landed at NY 03:xx the next calendar day).
    assert run_guard("schedule", "03", "2026-08-26", "2026-08-24", "17") == "true"


def test_catches_up_across_a_weekend_when_genuinely_behind_by_more_than_a_weekday():
    # Today is Tuesday 2026-09-01; the last good weekday baseline is Monday
    # 2026-08-31, so a last_updated of Friday 2026-08-28 (skipped Monday
    # entirely) is genuinely one full weekday behind -> catch up.
    assert run_guard("schedule", "03", "2026-09-01", "2026-08-28", "17") == "true"


def test_never_built_before_runs_immediately_regardless_of_hour():
    assert run_guard("schedule", "03", "2026-08-28", "") == "true"
