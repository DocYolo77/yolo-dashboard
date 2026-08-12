"""
Tests for scripts/dst_guard.sh — the New York time-zone guard behind
refresh_data.yml's daily build (Full-Universe spec point 18/R): only run
the dashboard build when local America/New_York time is 16:xx, except for
a manual workflow_dispatch, which always runs regardless of time.
Run with: pytest tests/ -v
"""

import subprocess
from pathlib import Path

SCRIPT = str(Path(__file__).parent.parent / "scripts" / "dst_guard.sh")


def run_guard(event_name, override_hour=None):
    args = ["bash", SCRIPT, event_name]
    if override_hour is not None:
        args.append(override_hour)
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_runs_at_16_local_ny_time():
    assert run_guard("schedule", "16") == "true"


def test_skips_at_15_local_ny_time_edt_early_cron():
    # EDT (UTC-4): the 20:05 UTC cron lands at 16:05 NY (runs); the
    # 21:05 UTC cron lands at 17:05 NY (must skip). This test covers the
    # "wrong slot is 17" half of that pair.
    assert run_guard("schedule", "17") == "false"


def test_skips_at_15_local_ny_time_est_early_cron():
    # EST (UTC-5): the 20:05 UTC cron lands at 15:05 NY (must skip); the
    # 21:05 UTC cron lands at 16:05 NY (runs).
    assert run_guard("schedule", "15") == "false"


def test_workflow_dispatch_always_runs_regardless_of_hour():
    for hour in ("00", "09", "15", "16", "17", "23"):
        assert run_guard("workflow_dispatch", hour) == "true"


def test_workflow_dispatch_ignores_hour_entirely_even_without_override():
    # No override -> would call the real `date` command, but workflow_dispatch
    # must short-circuit before ever reaching that branch.
    assert run_guard("workflow_dispatch") == "true"


def test_only_exact_hour_16_matches_not_adjacent_minutes_within_other_hours():
    for hour in ("14", "18", "20", "21"):
        assert run_guard("schedule", hour) == "false"
