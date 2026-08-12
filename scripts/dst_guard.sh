#!/usr/bin/env bash
# YOLO Dashboard — New York time-zone guard (Full-Universe spec point 18).
#
# GitHub Actions cron has no DST-aware timezones, so refresh_data.yml
# schedules TWO fixed UTC crons (one correct for EDT, one for EST) that both
# fire year-round. This script decides which of those two firings is the
# real one: it prints "true" only when the current wall-clock time in
# America/New_York is 16:xx (right after NYSE close), "false" otherwise.
# Deliberately checks the HOUR only, not minute 05 exactly — GitHub
# Scheduled Actions can start late.
#
# workflow_dispatch (manual trigger) ALWAYS returns "true", regardless of
# time — a human explicitly asking for a run should never be silently
# skipped by this guard.
#
# Usage: dst_guard.sh <event_name> [override_local_hour]
#   event_name:          GitHub Actions ${{ github.event_name }}
#   override_local_hour: optional 2-digit hour (00-23) — when given, skips
#                         the real `date` call entirely. Exists so this
#                         script's decision logic can be unit-tested without
#                         needing to fake the system clock/timezone
#                         (tests/test_dst_guard.py calls this directly).
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

if [ "$LOCAL_HOUR" = "16" ]; then
  echo "true"
else
  echo "false"
fi
