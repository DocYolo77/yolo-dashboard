"""Build a local, versioned Massive EOD snapshot for Dashboard 2.0."""

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time as clock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4
from zoneinfo import ZoneInfo


BENCHMARKS = {"SPY", "QQQ", "IWM", "RSP"}
STOCK_TYPES = {"CS", "ADRC", "ADRP", "ADRR", "ADRW"}
HISTORY = 460
CORRECTION_SESSIONS = 5
CHART_SESSIONS = 252
HOST = "api.massive.com"
SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]*\Z")


class DataError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Massive:
    def __init__(self, key, delay):
        self.key = key
        self.delay = delay
        self.last_request = -math.inf
        self.opener = build_opener(NoRedirect())

    def get(self, path, **params):
        url = f"https://{HOST}{path}?{urlencode(params)}" if path.startswith("/") else path
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != HOST:
            raise DataError("Unexpected Massive pagination host")
        request = Request(url, headers={"Authorization": f"Bearer {self.key}"})

        for attempt in range(4):
            clock.sleep(max(0, self.delay - (clock.monotonic() - self.last_request)))
            self.last_request = clock.monotonic()
            try:
                with self.opener.open(request, timeout=45) as response:
                    body = json.load(response)
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise DataError(f"Massive HTTP {error.code} at {parsed.path}") from None
                retry_after = error.headers.get("Retry-After", "")
                clock.sleep(int(retry_after) if retry_after.isdecimal() else 2 ** attempt)
            except (URLError, TimeoutError):
                if attempt == 3:
                    raise DataError(f"Massive request failed at {parsed.path}") from None
                clock.sleep(2 ** attempt)
            else:
                if body.get("status") not in {"OK", "DELAYED"}:
                    raise DataError(f"Massive rejected {parsed.path}")
                return body

    def pages(self, path, **params):
        seen = set()
        while path:
            if path in seen:
                raise DataError("Massive returned a repeated page")
            seen.add(path)
            page = self.get(path, **params)
            yield from page.get("results", [])
            path = page.get("next_url")
            params = {}

    def bars(self, session):
        return self.get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{session}",
            adjusted="false", include_otc="false",
        )

    def universe(self, session):
        return {
            row["ticker"]: row
            for row in self.pages(
                "/v3/reference/tickers", market="stocks", active="true",
                date=session, sort="ticker", limit=1000,
            )
            if row.get("type") in STOCK_TYPES and SYMBOL.fullmatch(row.get("ticker", ""))
        }

    def splits(self, start, end):
        return self.pages(
            "/stocks/v1/splits", limit=1000,
            **{"execution_date.gte": start, "execution_date.lte": end},
        )


def session_ready(calendar, session, slot):
    eastern = ZoneInfo("America/New_York")
    day = date.fromisoformat(session)
    local_slot = time(20, 30)
    if slot == "next-morning":
        day += timedelta(days=1)
        local_slot = time(0, 30)
    scheduled = datetime.combine(day, local_slot, eastern)
    return max(scheduled, calendar.session_close(session) + timedelta(minutes=30))


def select_sessions(calendar, now, slot, target=None):
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    if target is None:
        latest = calendar.date_to_session(today.isoformat(), direction="previous")
        target = latest.date().isoformat()
        while now < session_ready(calendar, target, slot):
            latest = calendar.previous_session(latest)
            target = latest.date().isoformat()
    if not calendar.is_session(target):
        raise DataError(f"{target} is not an XNYS session")
    if now < session_ready(calendar, target, slot):
        raise DataError(f"{target} is before the configured data slot")
    start = (date.fromisoformat(target) - timedelta(days=800)).isoformat()
    start = max(start, calendar.first_session.date().isoformat())
    dates = calendar.sessions_in_range(start, target)[-HISTORY:]
    if len(dates) != HISTORY:
        raise DataError(f"XNYS calendar provides fewer than {HISTORY} sessions through {target}")
    return [day.date().isoformat() for day in dates]


def valid_bar(row):
    o, h, low, c, v = (row.get(key) for key in ("o", "h", "l", "c", "v"))
    return (
        all(isinstance(x, (int, float)) and math.isfinite(x) for x in (o, h, low, c, v))
        and 0 < low <= min(o, c) <= max(o, c) <= h and v > 0
    )


def fetch_session(source, session, previous_count, min_rows):
    payload = source.bars(session)
    rows = payload.get("results", [])
    if payload.get("resultsCount", len(rows)) != len(rows):
        raise DataError(f"Truncated Massive response for {session}")
    bars = {}
    for row in rows:
        symbol = row.get("T")
        if not symbol or not valid_bar(row):
            continue
        if symbol in bars:
            raise DataError(f"Duplicate symbol in {session}")
        vw = row.get("vw")
        if not isinstance(vw, (int, float)) or not math.isfinite(vw) or vw <= 0:
            vw = None
        bars[symbol] = tuple(row[key] for key in ("o", "h", "l", "c", "v")) + (vw,)
    if not BENCHMARKS <= bars.keys() or len(bars) < max(min_rows, previous_count * 0.85):
        raise DataError(f"Incomplete Massive session {session}")
    return bars


def split_events(source, sessions):
    events = defaultdict(list)
    seen = set()
    for event in source.splits(sessions[0], sessions[-1]):
        symbol = event["ticker"]
        day = event["execution_date"]
        if not sessions[0] <= day <= sessions[-1]:
            raise DataError("Split outside requested date range")
        split_from = event["split_from"]
        split_to = event["split_to"]
        if split_from <= 0 or split_to <= 0:
            raise DataError("Invalid Massive split ratio")
        identity = event.get("id") or (symbol, day, split_from, split_to)
        if identity not in seen:
            events[symbol].append((day, split_from / split_to))
            seen.add(identity)
    return events


def adjust(row, events):
    session, o, h, low, c, volume, vw = row
    factor = math.prod(ratio for execution, ratio in events if session < execution)
    return session, o * factor, h * factor, low * factor, c * factor, volume / factor, (
        None if vw is None else vw * factor
    )


def json_write(path, data):
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        output.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def rolling_average(values, periods):
    return sum(values[-periods:]) / periods if len(values) >= periods else None


def chart_bars(bars):
    closes = []
    ema_num = [0.0, 0.0]
    ema_den = [0.0, 0.0]
    result = []
    for session, o, h, low, c, volume, _ in bars:
        closes.append(c)
        for index, periods in enumerate((10, 20)):
            decay = (periods - 1) / (periods + 1)
            ema_num[index] = c + decay * ema_num[index]
            ema_den[index] = 1 + decay * ema_den[index]
        result.append({
            "date": session, "o": o, "h": h, "l": low, "c": c, "v": volume,
            "ema10": ema_num[0] / ema_den[0], "ema20": ema_num[1] / ema_den[1],
            "sma50": rolling_average(closes, 50), "sma200": rolling_average(closes, 200),
        })
    return result[-CHART_SESSIONS:]


def screener_row(symbol, metadata, bars, sessions, last):
    by_date = {row[0]: row for row in bars}
    close = last["c"]

    def trailing(days, offset=0):
        stop = len(sessions) - offset
        return [by_date.get(day) for day in sessions[max(0, stop - days):stop]]

    range_window = trailing(20)
    volume_window = trailing(50, offset=1)
    adv = None
    adr = None
    rvol = None
    if len(range_window) == 20 and all(range_window):
        adr = sum((row[2] / row[3] - 1) * 100 for row in range_window) / 20
        if all(row[6] is not None for row in range_window):
            adv = sum(row[5] * row[6] for row in range_window) / 20
    if len(volume_window) == 50 and all(volume_window):
        rvol = last["v"] / (sum(row[5] for row in volume_window) / 50)
    returns = {}
    for days in (1, 5, 21, 63, 126):
        previous = by_date.get(sessions[-1 - days]) if len(sessions) > days else None
        returns[f"return_{days}d_pct"] = (close / previous[4] - 1) * 100 if previous else None
    return {
        "symbol": symbol, "name": metadata.get("name"), "exchange": metadata.get("primary_exchange"),
        "type": metadata.get("type"), "close": close, "volume": last["v"], "adr20": adr,
        "adtv20_usd": adv, "rvol50": rvol, **returns,
        **{key: last[key] for key in ("ema10", "ema20", "sma50", "sma200")},
    }


def stage_snapshot(db, directory, sessions, universe, splits):
    session = sessions[-1]
    chart_dir = directory / "charts"
    chart_dir.mkdir(parents=True)
    screen = []
    market = {}
    for symbol in sorted(universe.keys() | BENCHMARKS):
        raw = db.execute(
            "SELECT session,o,h,l,c,v,vw FROM bars WHERE symbol=? ORDER BY session", (symbol,)
        )
        bars = [adjust(row, splits.get(symbol, ())) for row in raw]
        if not bars or bars[-1][0] != session:
            continue
        chart = chart_bars(bars)
        json_write(chart_dir / f"{symbol}.json", {"as_of_date": session, "symbol": symbol, "bars": chart})
        if symbol in BENCHMARKS:
            market[symbol] = chart[-1]
        else:
            screen.append(screener_row(symbol, universe[symbol], bars, sessions, chart[-1]))
    if not screen or set(market) != BENCHMARKS:
        raise DataError("Snapshot missing screener rows or benchmarks")
    json_write(directory / "market.json", {"as_of_date": session, "benchmarks": market})
    json_write(directory / "screener.json", {"as_of_date": session, "rows": screen})
    return len(screen)


def build(root, source, sessions, *, min_rows=5000, correction_sessions=CORRECTION_SESSIONS):
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / "run_lock.sqlite3", timeout=5)) as lock:
        with lock:
            lock.execute("BEGIN IMMEDIATE")
            return build_locked(root, source, sessions, min_rows, correction_sessions)


def build_locked(root, source, sessions, min_rows, correction_sessions):
    target = sessions[-1]
    manifest_path = root / "latest.json"
    with closing(sqlite3.connect(root / "market_history.sqlite3", timeout=5)) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT, session TEXT, o REAL, h REAL, l REAL, c REAL, v REAL, vw REAL,
                PRIMARY KEY (symbol, session)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS bars_session ON bars(session);
            CREATE TABLE IF NOT EXISTS sessions (session TEXT PRIMARY KEY, row_count INTEGER NOT NULL);
        """)
        with db:
            db.execute("BEGIN IMMEDIATE")
            current = read_json(manifest_path)
            if current and current["as_of_date"] > target:
                raise DataError("This run would roll back the published session")
            existing = dict(db.execute("SELECT session,row_count FROM sessions"))
            if existing and max(existing) > target:
                raise DataError("This run would overwrite a newer price store")
            universe = source.universe(target)
            if len(universe) < min_rows // 2:
                raise DataError(f"Incomplete Massive ticker reference for {target}")
            corrections = set(sessions[-correction_sessions:]) if correction_sessions else set()
            prior_count = 0
            for session in sessions:
                if session not in existing or session in corrections:
                    bars = fetch_session(source, session, existing.get(session, prior_count), min_rows)
                    db.execute("DELETE FROM bars WHERE session=?", (session,))
                    db.executemany(
                        "INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)",
                        ((symbol, session, *bar) for symbol, bar in bars.items()),
                    )
                    db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)", (session, len(bars)))
                    prior_count = len(bars)
                else:
                    prior_count = existing[session]
            db.execute("DELETE FROM bars WHERE session < ?", (sessions[0],))
            db.execute("DELETE FROM sessions WHERE session < ?", (sessions[0],))
            splits = split_events(source, sessions)
            version = f"{target}-{uuid4().hex}"
            directory = root / "snapshots" / version
            try:
                row_count = stage_snapshot(db, directory, sessions, universe, splits)
                manifest = {
                    "schema_version": 1, "as_of_date": target, "version": version,
                    "generated_at": datetime.now(timezone.utc).isoformat(), "source": "Massive",
                    "market": f"snapshots/{version}/market.json",
                    "screener": f"snapshots/{version}/screener.json",
                    "charts": f"snapshots/{version}/charts/",
                    "rows": row_count, "history_sessions": len(sessions),
                }
                if current and current["as_of_date"] == target:
                    # Permit corrections but keep the public revision if every payload is identical.
                    prior = root / "snapshots" / current["version"]
                    paths = [path.relative_to(directory) for path in directory.rglob("*.json")]
                    if len(paths) == sum(1 for _ in prior.rglob("*.json")) and all(
                        (prior / path).is_file() and (prior / path).read_bytes() == (directory / path).read_bytes()
                        for path in paths
                    ):
                        shutil.rmtree(directory)
                        return current
                temp_manifest = root / f".latest-{version}.tmp"
                json_write(temp_manifest, manifest)
                # Keep readers on the prior version until the durable ingest
                # commits, with the separate run lock held across both steps.
                db.commit()
                os.replace(temp_manifest, manifest_path)
            except Exception:
                if read_json(manifest_path) == current:
                    shutil.rmtree(directory, ignore_errors=True)
                    (root / f".latest-{version}.tmp").unlink(missing_ok=True)
                raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, help="Private persistent output directory")
    parser.add_argument("--session", help="Explicit NY session for a historical replay")
    parser.add_argument("--slot", choices=("evening", "next-morning"), default="evening")
    parser.add_argument("--request-delay", type=float, default=0.2, help="Use at least 12.1 seconds on Basic")
    parser.add_argument("--probe", action="store_true", help="Check key access without writing market data")
    args = parser.parse_args()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        parser.error("MASSIVE_API_KEY is required")
    if not math.isfinite(args.request_delay) or args.request_delay < 0:
        parser.error("--request-delay must be finite and nonnegative")
    if not args.probe:
        if args.store is None:
            parser.error("--store is required for a snapshot")
        store = args.store.resolve()
        if store.is_relative_to(Path(__file__).resolve().parents[1]):
            parser.error("--store must be outside the Git checkout")
    try:
        import exchange_calendars

        calendar = exchange_calendars.get_calendar("XNYS")
        now = datetime.now(timezone.utc)
        source = Massive(key, args.request_delay)
        if args.probe:
            sessions = select_sessions(calendar, now, "next-morning")
            session = sessions[-1]
            fetch_session(source, session, 0, 5000)
            print(f"Key can read completed session {session}.")
            today = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
            if calendar.is_session(today) and today != session:
                try:
                    payload = source.bars(today)
                except DataError as error:
                    if not str(error).startswith("Massive HTTP 403"):
                        raise
                    print("Current NY date returned HTTP 403. Verify overnight access before selecting a slot.")
                else:
                    if payload.get("results"):
                        print("Current NY date has data. This does not establish EOD completeness.")
                    else:
                        print("Current NY date returned no bars at the time of this probe.")
            return 0
        sessions = select_sessions(calendar, now, args.slot, args.session)
        result = build(store, source, sessions)
    except (DataError, OSError, ValueError, sqlite3.Error) as error:
        print(f"EOD snapshot failed: {error}", file=sys.stderr)
        return 1
    print(f"EOD snapshot ready for {result['as_of_date']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
