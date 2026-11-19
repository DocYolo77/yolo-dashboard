from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

from scripts import build_eod_snapshot as eod


def sessions(count):
    current = date(2024, 1, 2)
    result = []
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


class FakeMassive:
    def __init__(self):
        self.received = []
        self.failed = None
        self.events = []
        self.after_split = None
        self.incomplete = None
        self.corrected = None

    def universe(self, session):
        return {"ACME": {"name": "Acme", "primary_exchange": "XNYS", "type": "CS"}}

    def splits(self, start, end):
        return self.events

    def bars(self, session):
        self.received.append(session)
        if session == self.failed:
            raise eod.DataError("simulated timeout")
        tickers = eod.BENCHMARKS | {"ACME"}
        if session == self.incomplete:
            tickers -= {"QQQ"}
        rows = []
        for symbol in sorted(tickers):
            price = 100 if symbol != "ACME" or not self.after_split or session < self.after_split else 50
            if symbol == "ACME" and session == self.corrected:
                price += 2
            rows.append({"T": symbol, "o": price, "h": price + 1, "l": price - 1,
                         "c": price, "v": 1000.25, "vw": price})
        return {"resultsCount": len(rows), "results": rows}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = FakeMassive()

    def run_build(self, dates):
        return eod.build(self.root, self.source, dates, min_rows=3)

    def chart(self, manifest):
        return eod.read_json(self.root / manifest["charts"] / "ACME.json")

    def test_five_consecutive_replays_share_date_and_use_warm_store(self):
        dates = sessions(464)
        for stop in range(460, 465):
            self.source.received.clear()
            expected = dates[stop - 1]
            manifest = self.run_build(dates[stop - 460:stop])
            self.assertEqual(expected, manifest["as_of_date"])
            market = eod.read_json(self.root / manifest["market"])
            screen = eod.read_json(self.root / manifest["screener"])
            chart = self.chart(manifest)
            self.assertEqual({expected}, {market["as_of_date"], screen["as_of_date"], chart["as_of_date"]})
            self.assertEqual(set(market["benchmarks"]), eod.BENCHMARKS)
            self.assertEqual(len(chart["bars"]), 252)
            self.assertEqual(chart["bars"][0]["sma200"], 100)
            self.assertEqual(chart["bars"][-1]["v"], 1000.25)
            self.assertEqual(screen["rows"][0]["close"], chart["bars"][-1]["c"])
            self.assertEqual(screen["rows"][0]["adtv20_usd"], 100025)
            self.assertEqual(len(self.source.received), 460 if stop == 460 else 5)

    def test_identical_replay_keeps_one_revision_and_changed_bar_revises(self):
        dates = sessions(6)
        first = self.run_build(dates)
        self.assertEqual(first, self.run_build(dates))
        self.assertEqual(len(list((self.root / "snapshots").iterdir())), 1)
        self.source.corrected = dates[-2]
        corrected = self.run_build(dates)
        self.assertNotEqual(first["version"], corrected["version"])
        self.assertEqual(self.chart(first)["bars"][-2]["c"], 100)
        self.assertEqual(self.chart(corrected)["bars"][-2]["c"], 102)

    def test_split_recomputes_prior_prices_and_preserves_dollar_volume(self):
        dates = sessions(22)
        first = self.run_build(dates[:-1])
        self.source.after_split = dates[-1]
        self.source.events = [{"ticker": "ACME", "execution_date": dates[-1],
                               "split_from": 1, "split_to": 2}]
        new = self.run_build(dates)
        old_bar, new_bar = self.chart(new)["bars"][-2:]
        self.assertEqual(old_bar["c"], new_bar["c"])
        self.assertEqual(old_bar["v"], 2 * new_bar["v"])
        self.assertEqual(self.chart(first)["bars"][-1]["c"], 100)
        screen = eod.read_json(self.root / new["screener"])
        self.assertEqual(screen["rows"][0]["return_1d_pct"], 0)

    def test_incomplete_or_timed_out_run_keeps_last_manifest_and_database(self):
        dates = sessions(6)
        first = self.run_build(dates[:-1])
        self.source.incomplete = dates[-1]
        with self.assertRaisesRegex(eod.DataError, "Incomplete"):
            self.run_build(dates)
        self.source.failed = dates[-1]
        with self.assertRaisesRegex(eod.DataError, "timeout"):
            self.run_build(dates)
        self.assertEqual(eod.read_json(self.root / "latest.json"), first)
        with closing(sqlite3.connect(self.root / "market_history.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT max(session) FROM sessions").fetchone()[0], dates[-2])

    def test_publish_failure_keeps_prior_revision_and_same_day_can_recover(self):
        dates = sessions(6)
        first = self.run_build(dates[:-1])
        with patch.object(eod.os, "replace", side_effect=OSError("simulated rename failure")):
            with self.assertRaises(OSError):
                self.run_build(dates)
        self.assertEqual(eod.read_json(self.root / "latest.json"), first)
        self.assertEqual(self.run_build(dates)["as_of_date"], dates[-1])

    def test_concurrent_same_date_produces_one_revision(self):
        dates = sessions(6)
        entered = Event()
        resume = Event()
        original = self.source.bars

        def gated(session):
            entered.set()
            if not resume.wait(3):
                raise AssertionError("test did not release first writer")
            return original(session)

        with patch.object(self.source, "bars", side_effect=gated):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.run_build, dates)
                self.assertTrue(entered.wait(3))
                second = pool.submit(self.run_build, dates)
                resume.set()
                self.assertEqual(first.result(timeout=5), second.result(timeout=5))
        self.assertEqual(len(list((self.root / "snapshots").iterdir())), 1)


class MassiveTests(unittest.TestCase):
    def test_403_fails_without_retry(self):
        source = eod.Massive("dummy", 0)
        failure = HTTPError("https://api.massive.com", 403, "blocked", {}, None)
        with patch.object(source.opener, "open", side_effect=failure) as request:
            with self.assertRaisesRegex(eod.DataError, "HTTP 403"):
                source.bars("2026-11-18")
        self.assertEqual(request.call_count, 1)

    def test_429_retries_but_stops_at_limit(self):
        source = eod.Massive("dummy", 0)
        failure = HTTPError("https://api.massive.com", 429, "limited", {"Retry-After": "0"}, None)
        with patch.object(source.opener, "open", side_effect=failure) as request:
            with self.assertRaisesRegex(eod.DataError, "HTTP 429"):
                source.bars("2026-11-18")
        self.assertEqual(request.call_count, 4)

    def test_rejects_non_provider_pagination_url(self):
        source = eod.Massive("dummy", 0)
        with self.assertRaisesRegex(eod.DataError, "pagination host"):
            source.get("https://outside.example/splits")


try:
    import exchange_calendars
except ModuleNotFoundError:
    exchange_calendars = None


@unittest.skipUnless(exchange_calendars, "install requirements-eod.txt for calendar integration")
class CalendarTests(unittest.TestCase):
    def test_regular_calendar_early_close_and_thanksgiving(self):
        calendar = exchange_calendars.get_calendar("XNYS")
        now = datetime(2026, 11, 28, 2, tzinfo=timezone.utc)
        dates = eod.select_sessions(calendar, now, "evening")
        self.assertEqual(len(dates), 460)
        self.assertEqual(dates[-1], "2026-11-27")
        self.assertNotIn("2026-11-26", dates)
        eastern = ZoneInfo("America/New_York")
        self.assertEqual(calendar.session_close("2026-11-27").astimezone(eastern).hour, 13)

    def test_dst_and_after_midnight_slot(self):
        calendar = exchange_calendars.get_calendar("XNYS")
        standard = eod.session_ready(calendar, "2026-03-06", "evening")
        daylight = eod.session_ready(calendar, "2026-03-09", "evening")
        self.assertEqual(standard.astimezone(timezone.utc).hour, 1)
        self.assertEqual(daylight.astimezone(timezone.utc).hour, 0)
        now = datetime(2026, 3, 10, 4, 20, tzinfo=timezone.utc)
        self.assertEqual(eod.select_sessions(calendar, now, "next-morning")[-1], "2026-03-06")
        with self.assertRaisesRegex(eod.DataError, "before the configured"):
            eod.select_sessions(calendar, now, "next-morning", target="2026-03-09")


if __name__ == "__main__":
    unittest.main()
