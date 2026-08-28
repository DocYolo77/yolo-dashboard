#!/usr/bin/env bash
# YOLO Dashboard — New York time-zone guard (Full-Universe spec point 18),
# made resilient against GitHub Actions scheduled-run delays: a real
# incident (2026-08-25/26) saw a scheduled firing execute 11+ hours late,
# landing well outside the intended 16:xx NY window on every subsequent
# firing for days -- the OLD exact "hour == 16" check silently skipped the
# actual build every time, with no fresh data committed at all in the
# meantime, even though every run reported "success" (skipped steps are not
# a failure).
#
# GitHub Actions cron has no DST-aware timezones, so refresh_data.yml still
# schedules TWO fixed UTC crons (one correct for EDT, one for EST) that both
# fire year-round. Decision order:
#   1. workflow_dispatch -> always "true" (manual trigger, never skipped).
#   2. Already refreshed today (NY calendar date) -> "false" -- this dedup
#      check, not hour-matching, is what now keeps only ONE of the two daily
#      cron firings from actually building on a normal day.
#   3. More than one full weekday behind (the last successful refresh
#      predates the most recent Mon-Fri before today, correctly skipping
#      weekends so a normal Monday's "last update = Friday" is NOT treated
#      as stale) -> "true" regardless of hour -- catch-up: once behind, run
#      on the next firing instead of waiting to land in the 16:xx window
#      again, which is exactly what kept failing during the incident.
#   4. Local NY hour >= 16 (widened from an exact "== 16" match, since a
#      firing delayed by a few hours within the SAME day should still
#      count) -> "true".
#   5. Otherwise -> "false" (too early in the trading day, not badly behind).
#
# Usage: dst_guard.sh <event_name> [override_local_hour] [override_today_ny] [override_last_updated_ny]
#   event_name:                GitHub Actions ${{ github.event_name }}
#   override_local_hour:       optional 2-digit hour (00-23); skips the real
#                               `date` call for the current NY hour.
#   override_today_ny:         optional YYYY-MM-DD; skips the real `date`
#                               call for "today" in NY.
#   override_last_updated_ny:  optional YYYY-MM-DD (or "" for "never built
#                               yet"); supplying this argument AT ALL (even
#                               empty) skips reading data/dashboard_state.json
#                               entirely.
# All three overrides exist so this script's decision logic can be unit-
# tested deterministically (tests/test_dst_guard.py calls this directly)
# without faking the system clock or checking out real data files.
set -euo pipefail

EVENT_NAME="${1:-}"

if [ "$EVENT_NAME" = "workflow_dispatch" ]; then
  echo "true"
  exit 0
fi

if [ -n "${2:-}" ]; then
  LOCAL_HOUR="$2"
else
  LOCAL_HOUR=$(TZ=America/New_York date +%H)
fi
# Force base-10: `[ -ge ]` evaluates operands the same way arithmetic
# contexts do, where a leading zero means octal -- "08"/"09" would
# otherwise error out ("value too great for base").
LOCAL_HOUR=$((10#$LOCAL_HOUR))

if [ -n "${3:-}" ]; then
  TODAY_NY="$3"
else
  TODAY_NY=$(TZ=America/New_York date +%F)
fi

if [ "${4+set}" = "set" ]; then
  LAST_UPDATED_NY="$4"
else
  LAST_UPDATED_NY=""
  if [ -f data/dashboard_state.json ]; then
    LAST_UPDATED_UTC=$(python3 -c "
import json
try:
    with open('data/dashboard_state.json') as f:
        print(json.load(f).get('meta', {}).get('updated_at', ''))
except Exception:
    print('')
" 2>/dev/null || true)
    if [ -n "$LAST_UPDATED_UTC" ]; then
      LAST_UPDATED_NY=$(TZ=America/New_York date -d "$LAST_UPDATED_UTC" +%F 2>/dev/null || true)
    fi
  fi
fi

if [ "$LAST_UPDATED_NY" = "$TODAY_NY" ]; then
  echo "false"
  exit 0
fi

# Most recent Mon-Fri strictly before TODAY_NY.
DOW=$(date -d "$TODAY_NY" +%u)
case "$DOW" in
  1) BACK_DAYS=3 ;;  # Monday -> last Friday
  6) BACK_DAYS=1 ;;  # Saturday -> Friday
  7) BACK_DAYS=2 ;;  # Sunday -> Friday
  *) BACK_DAYS=1 ;;  # Tue-Fri -> yesterday
esac
PREV_WEEKDAY=$(date -d "$TODAY_NY -$BACK_DAYS days" +%F)

if [ -z "$LAST_UPDATED_NY" ] || [[ "$LAST_UPDATED_NY" < "$PREV_WEEKDAY" ]]; then
  echo "true"
  exit 0
fi

if [ "$LOCAL_HOUR" -ge 16 ]; then
  echo "true"
else
  echo "false"
fi
