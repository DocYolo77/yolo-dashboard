#!/usr/bin/env python3
"""
YOLO Dashboard — Full-Market Feature Engine
Computes the daily quantitative feature set for the ENTIRE eligible US stock
universe (not just the curated narrative baskets): price, market cap,
ADR20, eligibility, 1D/1W/1M performance, Full-Market RS percentiles,
per-ticker Thrust, EMA10/EMA20 + distance, ATR/ATR-Extension, and a
deterministic Discovery-Candidate flag. Writes data/market_features.json.

Design notes (see also the technical report handed to the user):

- Market-wide price history comes from ONE grouped-daily call per trading
  day (same pattern as build_narratives.py's fetch_grouped_history) — cheap
  regardless of how many tickers are in the universe.
- Market cap is NOT available in bulk from Massive; it requires the
  per-ticker overview endpoint. To keep the daily run's API budget real
  (not just architecturally "nice"), market-cap enrichment is requested
  ONLY for tickers that already pass the cheap type+ADR20 screen, and the
  result is cached for up to a week (see build_market_reference.py).
- Canonical formulas reused from elsewhere in this repo rather than
  reinvented:
    * EMA10/EMA20  -> same as scripts/build_data.py:calc_moving_averages
                       (close.ewm(span=n).mean(), pandas default adjust=True).
                       NOTE: scripts/build_data.py's QQQ-breadth code uses
                       ewm(..., adjust=False) for a *different* purpose
                       (breadth-of-market EMAs). Two conventions already
                       existed in this repo before this change; this script
                       follows calc_moving_averages because that is the
                       function whose output is literally labelled
                       "EMA10"/"EMA20" on the dashboard today.
    * RS percentile -> build_narratives.percentile_ranks(), unchanged,
                       just fed the eligible universe instead of the
                       118-ticker curated set. CORRECTED (V1 dashboard
                       rebuild, point 7): percentile_ranks() is now called
                       AFTER eligibility (ADR20 + market cap) is resolved,
                       and only on the eligible subset — previously it
                       ranked against every ticker with a computable price
                       feature (~5x larger pool, includes sub-$1B/low-ADR
                       names), which understated true Full-Market RS for
                       eligible names. Same fix applies to Thrust
                       percentiles (thrust_percentile_1d/1w/1m, new fields
                       needed by the Opportunity Engine's Fresh Leader
                       rule). Non-eligible tickers get percentile = None,
                       not 0 — they are excluded from ranking, not ranked
                       at the bottom.
    * Thrust         -> same EMA(short)-EMA(long)-of-daily-return-series
                       shape as build_narratives.calc_basket_scores' basket
                       Thrust, applied to a single ticker's own daily return
                       series instead of a basket's median return series.
  ATR itself has NO prior definition anywhere in this repo (grep confirmed):
    * ATR(14) = simple 14-day moving average of True Range
                (True Range = max(H-L, |H-PrevClose|, |L-PrevClose|)).
                A plain SMA (not Wilder smoothing) was chosen so the value
                is reproducible from a fixed window without carrying hidden
                recursive state across runs.
  ATR Extension is the user-supplied canonical trading-system definition
  (extension from the 50-day MA, expressed in ATR-percent multiples, NOT
  the EMA10/dollar-ATR placeholder this script used before):
    * A = ATR%        = ATR($) / Last Price
    * B = %Gain-50MA   = (Close - SMA50) / SMA50
    * ATR Extension    = B / A
  i.e. how many "ATR percent" units price has run above (or below) its
  50-day simple moving average. SMA50 needs 50 trading days of history.

V1.1 (Structural Leadership & Narrative Engine) additions:
- TRADING_DAYS_NEEDED raised to 260 sessions (12M return + SMA50-slope
  buffer), fetched incrementally via a persistent, git-ignored rolling-
  window cache (see load_price_cache/save_price_cache/
  fetch_grouped_history_full_market_cached) instead of a full walk-back
  every run.
- structural_rs (0-100): multi-timeframe RS percentile (1M/3M/6M/12M),
  the primary structural-quality metric — replaces short-term RS/Thrust
  as the leadership gate.
- trend_strength (0-100): SMA50 slope percentile + persistence, i.e. trend
  QUALITY. Deliberately excludes distance/extension from SMA50 (that is
  Location, tracked separately via sma50_distance_pct/atr_extension).
- bootstrap_recent_leader: vectorized reconstruction of "was this ticker a
  Leader at any point in the last N sessions", replayed from the same
  260-session history so Recent Leader works from day one instead of
  waiting on daily-history hysteresis to accumulate.

V6 (Navigation/Universe-Filter/Screener/Narrative Strength synthesis)
additions -- see config/narrative_engine.json's _v1_note for the full
cross-file changelog:
- Deterministic Healthcare/Biotech universe exclusion (point 5-6), upstream
  of eligible=true and thus of ALL stock RS/Thrust percentiles: SIC-code-
  first classification (classify_healthcare_biotech), never a daily LLM
  call, never a single generic "health" keyword match. compute_eligibility/
  compute_eligible_universe gained an additive healthcare_excluded
  parameter (default False, so every pre-V6 caller is unaffected). Per-
  ticker audit fields healthcare_biotech_excluded/healthcare_exclusion_
  reason and meta counts healthcare_biotech_excluded_count/eligible_count_
  before_after_healthcare_exclusion are new in market_features.json.
- RSP (S&P 500 Equal Weight ETF, config rsp_benchmark.ticker) is folded
  into the SAME grouped-daily walk as the stock universe (zero extra API
  calls) and persisted into the shared price-history cache despite being
  an ETF -- exact same pattern build_narratives.py already used for SPY --
  then popped back out of `features` immediately so it can NEVER reach
  eligibility, RS/Thrust percentiles, or market_features.json's tickers
  block. RSP is a benchmark for build_narratives.py's new RSP-based
  Narrative Strength/Thrust and the visible Benchmark chart, never a stock.
- EMA21/ema21_distance_pct: purely ADDITIVE new fields for the Weekly
  Strength Screener preset (config screener_presets.weekly_strength). The
  pre-existing, dashboard-canonical EMA20 (used everywhere else, including
  the Opportunity Engine's near_emas flag) is completely unchanged.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_narratives import percentile_ranks, HORIZONS, compute_narrative_thrust_rsp  # noqa: E402  (reuse canonical logic)
import build_market_reference as ref  # noqa: E402

MASSIVE_BASE = "https://api.massive.com"
TRADING_DAYS_NEEDED = 260  # V1.1: 12M return (252 sessions) + buffer for SMA50-20-sessions-ago etc.
MAX_CALENDAR_LOOKBACK = 420  # must cover a cold-cache 260-trading-day bootstrap (~365-390 calendar days)
MIN_RESULTS_FOR_TRADING_DAY = 1000
ADR_LOOKBACK_DEFAULT = 20
PRICE_CACHE_SCHEMA_VERSION = 2  # v2: adds per-ticker "open" (scripts/build_ticker_charts.py candlesticks)


def massive_get(path, params=None, retries=3):
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        print("FATAL: MASSIVE_API_KEY nicht gesetzt.", file=sys.stderr)
        sys.exit(1)
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{MASSIVE_BASE}{path}"
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
    print(f"  ⚠ Request fehlgeschlagen ({path}): {last_err}", file=sys.stderr)
    return None


def load_config(path="config/narrative_engine.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_types_reference(path):
    p = Path(path)
    if not p.exists():
        print(f"  ⚠ {path} fehlt — hole Ticker-Typen-Liste einmalig (billig, ~10-30 Requests)...")
        return ref.build_types(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def type_eligible_universe(types_ref, excluded_types):
    """Tickers whose Massive `type` is NOT in the excluded set (ETF/ETN/FUND/...)."""
    universe = set()
    for sym, meta in types_ref["tickers"].items():
        t = meta.get("type")
        if t and t not in excluded_types:
            universe.add(sym)
    return universe


# ─────────────────────────────────────────────
# Market-wide grouped-daily price history
# ─────────────────────────────────────────────

def fetch_ticker_range_backfill(ticker, start_date, end_date):
    """One-time single-ticker historical backfill (spec point 9's explicit
    fallback: "if the current cache structure doesn't cleanly allow this, a
    small separate benchmark load may be built instead"). Unlike a genuine
    new IPO or a stock newly passing the type/ADR filter -- which correctly
    has no earlier history to backfill and is left to accumulate one day at
    a time -- a ticker like RSP already has decades of real trading history
    available from Massive; its absence from the shared cache is purely an
    artifact of when it was first added to the fetch universe, not a lack
    of real data. Left unfixed, the grouped-daily incremental walk (which
    only ever fetches days newer than the cache's existing coverage) would
    otherwise strand it at 1-2 days of history for ~260 daily runs before
    ANY Strength/Thrust window could ever populate.

    ONE extra API call (Massive/Polygon-style single-ticker range
    aggregates), only ever made when the ticker is genuinely missing from
    an already-populated cache -- a recurring daily run with RSP already in
    the cache never reaches this function again.
    Returns {date_str: {"o":..,"h":..,"l":..,"c":..}}."""
    data = massive_get(f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}",
                        params={"adjusted": "true", "sort": "asc", "limit": 50000})
    if not data or not data.get("results"):
        return {}
    out = {}
    for row in data["results"]:
        if row.get("t") is None:
            continue
        date_str = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out[date_str] = {"o": row.get("o"), "h": row.get("h"), "l": row.get("l"), "c": row.get("c")}
    return out


def load_price_cache(cache_path):
    """Load the persisted rolling-window price cache (V1.1 point 2). Returns
    None if absent or schema-mismatched (treated as a cold cache — the
    incremental fetch below transparently falls back to a full bootstrap)."""
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if cache.get("schema_version") != PRICE_CACHE_SCHEMA_VERSION:
        print(f"  ⚠ Preis-Cache-Schema veraltet (gefunden {cache.get('schema_version')}, "
              f"erwartet {PRICE_CACHE_SCHEMA_VERSION}) — verwerfe Cache, baue neu auf")
        return None
    return cache


def save_price_cache(cache_path, trading_days, per_ticker_close, per_ticker_high, per_ticker_low,
                      per_ticker_open, universe_set):
    """Persist the rolling window as {symbol: {"open":[...], "close":[...],
    "high":[...], "low":[...]}} aligned 1:1 to `trading_days`. Not pretty-
    printed (pure build artifact, never meant to be read by a human or
    diffed in git — it lives outside git entirely, see refresh_data.yml's
    actions/cache step). "open" (schema v2) exists purely for
    scripts/build_ticker_charts.py's hover-candlestick chart data — no
    existing feature calculation in this file needs it."""
    tickers_data = {}
    for sym in universe_set:
        close_series = [per_ticker_close.get(sym, {}).get(d) for d in trading_days]
        if not any(v is not None for v in close_series):
            continue  # nothing ever observed for this symbol -> don't bloat the cache with an all-null row
        tickers_data[sym] = {
            "open": [per_ticker_open.get(sym, {}).get(d) for d in trading_days],
            "close": close_series,
            "high": [per_ticker_high.get(sym, {}).get(d) for d in trading_days],
            "low": [per_ticker_low.get(sym, {}).get(d) for d in trading_days],
        }
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema_version": PRICE_CACHE_SCHEMA_VERSION, "dates": trading_days, "tickers": tickers_data}, f)


def fetch_grouped_history_full_market_cached(universe_set, cache_path, target_days=TRADING_DAYS_NEEDED,
                                              backfill_tickers=None):
    """Incremental, cache-backed version of the market-wide grouped-daily
    walk (V1.1 point 2). Loads whatever rolling window is already persisted
    at `cache_path` (populated across CI runs via GitHub Actions' built-in
    `actions/cache` — see refresh_data.yml; NOT committed to git, a 260-
    session x ~5500-ticker window would bloat repo history unboundedly),
    fetches ONLY the trading day(s) newer than the cache's newest entry
    (normally just one — yesterday's close), appends, trims to the most
    recent `target_days` sessions, and re-persists.

    On a cold cache (first run, or an evicted/expired GH Actions cache
    entry) this transparently performs the one-time full `target_days`
    walk-back the spec explicitly accepts as a bootstrap cost. New tickers
    not yet in the cache (recent IPOs, tickers newly passing the type
    filter) simply accumulate history going forward — same graceful
    partial-history handling calc_ticker_features() already applies
    (nothing here requires every ticker to have the full window).

    `backfill_tickers` (V6 point 9): tickers that, unlike a genuine new IPO,
    already have a long real trading history available from Massive even
    though they're new to THIS cache (e.g. RSP, added to the fetch universe
    by this very feature) — see fetch_ticker_range_backfill's docstring for
    why these need a one-time single-ticker catch-up instead of being left
    to accumulate one day at a time like an actual new listing.

    One API call covers the whole market per trading day regardless of
    universe size, same pattern as before this change.
    Returns (close_df, high_df, low_df, trading_days), trading_days ascending.
    """
    cache = load_price_cache(cache_path)
    per_ticker_close, per_ticker_high, per_ticker_low, per_ticker_open = {}, {}, {}, {}
    cached_dates = []

    if cache:
        cached_dates = cache["dates"]
        cached_tickers = cache["tickers"]
        for sym, series in cached_tickers.items():
            if sym not in universe_set:
                continue  # fell out of the universe (delisted/type change) -> drop, don't carry forward
            per_ticker_close[sym] = dict(zip(cached_dates, series["close"]))
            per_ticker_high[sym] = dict(zip(cached_dates, series["high"]))
            per_ticker_low[sym] = dict(zip(cached_dates, series["low"]))
            per_ticker_open[sym] = dict(zip(cached_dates, series.get("open", [])))
        n_missing = sum(1 for s in universe_set if s not in cached_tickers)
        print(f"  ✅ Preis-Cache geladen: {len(cached_dates)} Handelstage, {len(cached_tickers)} Ticker "
              f"({n_missing} neue Ticker im Universe noch nicht im Cache)")

        for ticker in (backfill_tickers or ()):
            if ticker not in universe_set or not cached_dates:
                continue  # not requested this run -> normal incremental path
            # Presence alone isn't enough: a ticker can already be a key in
            # cached_tickers while its "close" series is almost entirely
            # None (e.g. RSP was auto-added to per_ticker_close by a run
            # BEFORE this backfill existed, so it "has cache history" in
            # name only -- 2 real days out of 260 slots). Compare actual
            # non-null coverage against the cache's own date depth instead
            # of just checking dict membership, so a previously-stranded
            # ticker still gets backfilled exactly once.
            existing_close = cached_tickers.get(ticker, {}).get("close", [])
            existing_count = sum(1 for v in existing_close if v is not None)
            if existing_count >= len(cached_dates):
                continue  # already fully covers the cache's date range -> normal incremental path
            print(f"  📈 {ticker} hat nur {existing_count}/{len(cached_dates)} Handelstage im Preis-Cache "
                  f"— einmaliger Backfill via Einzel-Ticker-Range-Call "
                  f"({cached_dates[0]}..{cached_dates[-1]})")
            backfilled = fetch_ticker_range_backfill(ticker, cached_dates[0], cached_dates[-1])
            per_ticker_close.setdefault(ticker, {})
            per_ticker_high.setdefault(ticker, {})
            per_ticker_low.setdefault(ticker, {})
            per_ticker_open.setdefault(ticker, {})
            cached_dates_set = set(cached_dates)
            for date_str, ohlc in backfilled.items():
                if date_str not in cached_dates_set:
                    continue  # stay within this cache's own date set, never extend it independently
                per_ticker_close[ticker][date_str] = ohlc["c"]
                per_ticker_high[ticker][date_str] = ohlc["h"]
                per_ticker_low[ticker][date_str] = ohlc["l"]
                per_ticker_open[ticker][date_str] = ohlc["o"]
            print(f"  ✅ {ticker} Backfill: {len(backfilled)} Handelstage nachgeladen "
                  f"(1 zusaetzlicher API-Call, einmalig)")
    else:
        print("  ⚠ Kein (gueltiger) Preis-Cache gefunden — einmaliger Full-Bootstrap "
              f"({target_days} Handelstage)")

    for t in universe_set:
        per_ticker_close.setdefault(t, {})
        per_ticker_high.setdefault(t, {})
        per_ticker_low.setdefault(t, {})
        per_ticker_open.setdefault(t, {})

    newest_cached = cached_dates[-1] if cached_dates else None
    print(f"\n📊 Ergaenze marktweite Grouped-Daily-OHLC (Ziel: {target_days} Handelstage, "
          f"{len(cached_dates)} bereits vorhanden)...")

    new_days = []
    day = datetime.now(timezone.utc).date()
    calendar_checked = 0

    while calendar_checked < MAX_CALENDAR_LOOKBACK:
        date_str = day.isoformat()
        if newest_cached is not None and date_str <= newest_cached:
            break  # reached the cache's existing coverage -> nothing more to fetch
        if newest_cached is None and len(new_days) >= target_days:
            break  # cold-cache bootstrap complete

        data = massive_get(f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}")
        calendar_checked += 1
        day -= timedelta(days=1)

        if not data or data.get("resultsCount", 0) < MIN_RESULTS_FOR_TRADING_DAY:
            continue

        new_days.append(date_str)
        for row in data.get("results", []):
            sym = row.get("T")
            if sym in per_ticker_close:
                per_ticker_close[sym][date_str] = row.get("c")
                per_ticker_high[sym][date_str] = row.get("h")
                per_ticker_low[sym][date_str] = row.get("l")
                per_ticker_open[sym][date_str] = row.get("o")
        print(f"  → {date_str}: {data['resultsCount']} Ticker (neuer Handelstag {len(new_days)})")

    trading_days = sorted(set(cached_dates) | set(new_days))
    if len(trading_days) > target_days:
        trading_days = trading_days[-target_days:]  # roll the window

    print(f"  ✅ {len(trading_days)} Handelstage gesamt ({len(new_days)} neu geladen via API, "
          f"{calendar_checked} Kalendertage geprueft)")

    save_price_cache(cache_path, trading_days, per_ticker_close, per_ticker_high, per_ticker_low,
                      per_ticker_open, universe_set)

    close_df = build_frame(per_ticker_close, trading_days, universe_set)
    high_df = build_frame(per_ticker_high, trading_days, universe_set)
    low_df = build_frame(per_ticker_low, trading_days, universe_set)
    return close_df, high_df, low_df, trading_days


def build_frame(per_ticker, trading_days, universe_set):
    df = pd.DataFrame(index=trading_days, columns=sorted(universe_set), dtype=float)
    for sym, series in per_ticker.items():
        for date_str, val in series.items():
            if date_str in df.index:
                df.at[date_str, sym] = val
    return df


# ─────────────────────────────────────────────
# Per-ticker feature calculation
# ─────────────────────────────────────────────

def calc_true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def calc_sma50_trend_fields(close, sma50_series, slope_lookback, persistence_lookback):
    """SMA50 as a trend-strength anchor, not just an above/below flag (V1.1
    point 5). Returns (sma50_slope_20d_pct, pct_sessions_above_sma50_20d) —
    either may be None if there isn't enough history yet (graceful
    degradation, same as every other lookback-dependent field here)."""
    sma50_slope_pct = None
    if len(sma50_series) > slope_lookback:
        sma50_today = sma50_series.iloc[-1]
        sma50_ago = sma50_series.iloc[-1 - slope_lookback]
        if sma50_today and sma50_ago and not np.isnan(sma50_today) and not np.isnan(sma50_ago) and sma50_ago != 0:
            sma50_slope_pct = round(float((sma50_today / sma50_ago - 1) * 100), 2)

    pct_above = None
    above = (close > sma50_series)
    window = above.iloc[-persistence_lookback:]
    if len(window) >= persistence_lookback and sma50_series.iloc[-persistence_lookback:].notna().all():
        pct_above = round(float(window.mean() * 100), 1)

    return sma50_slope_pct, pct_above


def calc_ticker_features(close_df, high_df, low_df, adr_lookback,
                          sma50_slope_lookback=20, sma50_persistence_lookback=20):
    """Returns {symbol: {...}} with price/ADR20/EMA/ATR/performance fields,
    plus (V1.1) multi-timeframe returns and SMA50 trend-strength fields."""
    out = {}
    min_history = max(adr_lookback, 51 + max(sma50_slope_lookback, sma50_persistence_lookback))
    for sym in close_df.columns:
        close = close_df[sym].dropna()
        if len(close) < min_history:
            continue
        high = high_df[sym].reindex(close.index)
        low = low_df[sym].reindex(close.index)
        last = close.iloc[-1]

        def pct_ago(n):
            if len(close) > n:
                base = close.iloc[-1 - n]
                return round(float((last - base) / base * 100), 2) if base else None
            return None

        # ADR20 = mean of daily (High/Low - 1) * 100 over the lookback window.
        # Standard "Average Daily Range %" (Minervini/Qullamaggie-style),
        # deliberately High/Low-ratio based (not True-Range/prev-close based)
        # so it measures pure intraday range independent of gaps.
        daily_range_pct = (high / low - 1.0) * 100.0
        adr_window = daily_range_pct.dropna().iloc[-adr_lookback:]
        adr20 = round(float(adr_window.mean()), 2) if len(adr_window) >= adr_lookback else None

        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        # V6 point 21: EMA21 is ADDITIVE only, for the Weekly Strength
        # screener preset — the dashboard-canonical EMA20 (used everywhere
        # else, incl. near_emas) is completely unchanged.
        ema21 = close.ewm(span=21).mean().iloc[-1]
        ema10_distance_pct = round(float((last - ema10) / ema10 * 100), 2)
        ema20_distance_pct = round(float((last - ema20) / ema20 * 100), 2)
        ema21_distance_pct = round(float((last - ema21) / ema21 * 100), 2)
        sma50_series = close.rolling(50).mean()
        sma50 = sma50_series.iloc[-1]

        tr = calc_true_range(high, low, close)
        atr14 = tr.dropna().iloc[-14:].mean() if tr.dropna().shape[0] >= 14 else None
        atr = round(float(atr14), 4) if atr14 and atr14 > 0 else None

        # ATR Extension — user-supplied canonical formula (see module docstring):
        #   A = ATR% = ATR($) / Last Price
        #   B = %Gain-50MA = (Close - SMA50) / SMA50
        #   ATR Extension = B / A
        atr_extension = None
        gain_from_sma50_pct = None
        if sma50 and sma50 > 0:
            gain_from_sma50_pct = round(float((last - sma50) / sma50 * 100), 2)
        if atr and atr > 0 and last > 0 and gain_from_sma50_pct is not None:
            atr_pct = atr / last * 100.0
            atr_extension = round(float(gain_from_sma50_pct / atr_pct), 2)

        sma50_slope_20d_pct, pct_sessions_above_sma50_20d = calc_sma50_trend_fields(
            close, sma50_series, sma50_slope_lookback, sma50_persistence_lookback)

        out[sym] = {
            "symbol": sym,
            "close": round(float(last), 2),
            "adr20": adr20,
            "sma50": round(float(sma50), 2) if sma50 and not np.isnan(sma50) else None,
            "gain_from_sma50_pct": gain_from_sma50_pct,
            "sma50_distance_pct": gain_from_sma50_pct,  # V1.1 alias, same value (see report point 4)
            "d1_pct": pct_ago(1),
            "w1_pct": pct_ago(5),
            "m1_pct": pct_ago(21),
            "return_3m": pct_ago(63),
            "return_6m": pct_ago(126),
            "return_12m": pct_ago(252),
            "ema10": round(float(ema10), 2),
            "ema20": round(float(ema20), 2),
            "ema21": round(float(ema21), 2),
            "ema10_distance_pct": ema10_distance_pct,
            "ema20_distance_pct": ema20_distance_pct,
            "ema21_distance_pct": ema21_distance_pct,
            "atr": atr,
            "atr_extension": atr_extension,
            "sma50_slope_20d_pct": sma50_slope_20d_pct,
            "pct_sessions_above_sma50_20d": pct_sessions_above_sma50_20d,
        }
    return out


def calc_thrust(daily_ret_by_symbol, symbol, short, long):
    """Same shape as build_narratives.calc_basket_scores' Thrust, applied to
    one ticker's own daily-return series instead of a basket median series."""
    s = daily_ret_by_symbol.get(symbol)
    if s is None or s.dropna().shape[0] < long:
        return None
    ema_short = s.ewm(span=short, adjust=False).mean()
    ema_long = s.ewm(span=long, adjust=False).mean()
    return round(float(ema_short.iloc[-1] - ema_long.iloc[-1]), 2)


def compute_eligibility(adr20, market_cap, universe_cfg, healthcare_excluded=False):
    """Universe-Filter (point 2, extended V6 point 5): ADR20 > adr_minimum_pct
    AND market_cap >= market_cap_minimum_usd AND NOT healthcare_excluded.
    ADR/market_cap must be known (not None) — missing data is NOT eligible-
    by-default. healthcare_excluded defaults to False so pre-V6 callers
    (tests, other scripts) keep working unchanged. Extracted as a pure
    function so it is testable without a market-wide fetch."""
    adr_ok = adr20 is not None and adr20 > universe_cfg["adr_minimum_pct"]
    cap_ok = market_cap is not None and market_cap >= universe_cfg["market_cap_minimum_usd"]
    return bool(adr_ok and cap_ok and not healthcare_excluded)


def compute_eligible_universe(features, market_cap_by_symbol, universe_cfg, healthcare_excluded_by_symbol=None):
    """Resolve eligibility for every ticker with a computed feature set.
    MUST run before any percentile ranking (V1 rebuild point 7 fix, extended
    V6 point 5: the Healthcare/Biotech exclusion is upstream of eligible=true
    for the exact same reason — RS/Thrust percentiles must be ranked only
    against tickers that survive EVERY universe filter, not just ADR/cap)."""
    healthcare_excluded_by_symbol = healthcare_excluded_by_symbol or {}
    return {sym: compute_eligibility(f["adr20"], market_cap_by_symbol.get(sym), universe_cfg,
                                      healthcare_excluded_by_symbol.get(sym, False))
            for sym, f in features.items()}


# ─────────────────────────────────────────────
# V6: deterministic Healthcare/Biotech universe exclusion (point 5-6)
# ─────────────────────────────────────────────

def classify_healthcare_biotech(sic_code, sic_description, company_description, filter_cfg):
    """Deterministic, SIC-code-first Healthcare/Biotech classification.
    Returns (excluded: bool, reason: str|None).

    Priority order (never mixed):
      1. sic_code present -> checked against sic_codes/sic_prefixes, then
         (backstop) sic_description against sic_description_keywords.
      2. V6.1 (Narrative Ranking & UI Bugfix Patch, point 18): if the SIC
         was present but didn't map via either of the above, company_
         description gets ONE more check against the SAME high-precision
         phrase list used below for the no-SIC-at-all case -- V6 skipped
         this entirely whenever a SIC existed, which let real Biotech/
         Healthcare companies with an unmapped/generic SIC still count as
         eligible (e.g. an active 'Biotech' narrative surviving on the
         dashboard). Still deliberately narrow, company-self-describing
         noun phrases ('biotech company', 'hospital operator', ...), never
         a bare 'health' substring scan -- a normal tech company whose
         company_description merely mentions healthcare as one of several
         customer verticals must never match (spec point 6/18).
      3. sic_code missing entirely -> the same company_description_keywords
         check, independent of step 2."""
    sic_str = str(sic_code) if sic_code else None
    if sic_str:
        if sic_str in set(filter_cfg["sic_codes"]):
            return True, f"SIC {sic_str} in exclude list"
        for prefix in filter_cfg["sic_prefixes"]:
            if sic_str.startswith(prefix):
                return True, f"SIC {sic_str} matches prefix '{prefix}'"
        desc = (sic_description or "").lower()
        for kw in filter_cfg["sic_description_keywords"]:
            if kw.lower() in desc:
                return True, f"sic_description contains '{kw}' (SIC {sic_str})"
        cdesc = (company_description or "").lower()
        for kw in filter_cfg["company_description_keywords"]:
            if kw.lower() in cdesc:
                return True, f"SIC {sic_str} not mapped; company_description contains '{kw}'"
        return False, None

    desc = (company_description or "").lower()
    for kw in filter_cfg["company_description_keywords"]:
        if kw.lower() in desc:
            return True, f"no SIC code; company_description contains '{kw}'"
    return False, None


def compute_healthcare_excluded_universe(features, sic_by_symbol, universe_cfg):
    """{symbol: (excluded, reason)} for every ticker with a computed feature
    set, using whatever SIC/description data is already available (from the
    per-ticker reference enrichment — no extra API calls). Disabled entirely
    via universe.exclude_healthcare_biotech=false (returns all-False)."""
    if not universe_cfg.get("exclude_healthcare_biotech", False):
        return {sym: (False, None) for sym in features}
    filter_cfg = universe_cfg["healthcare_biotech_filter"]
    out = {}
    for sym in features:
        meta = sic_by_symbol.get(sym, {})
        out[sym] = classify_healthcare_biotech(
            meta.get("sic_code"), meta.get("sic_description"), meta.get("description"), filter_cfg)
    return out


def calc_return_n_sessions_ago(close_df, sym, window, sessions_ago):
    """V6.1 point 16: `window`-session % return of `sym`, ending `sessions_ago`
    real trading sessions before the most recent one available in close_df's
    own (dropna'd) series for that ticker -- used to reconstruct a historical
    Stock RS1W reference point for stock_thrust_rs, straight from the same
    260-session price history everything else in this file already uses.
    None if the ticker isn't in close_df or there isn't enough history for a
    full window at that reference point (graceful degradation, never a
    partial/padded window)."""
    if sym not in close_df.columns:
        return None
    s = close_df[sym].dropna()
    n = len(s)
    end = n - sessions_ago
    if end - 1 - window < 0:
        return None
    base = s.iloc[end - 1 - window]
    if not base:
        return None
    return float((s.iloc[end - 1] - base) / base * 100)


def eligible_percentile_ranks(features, eligible_by_symbol, field):
    """percentile_ranks() restricted to the eligible subset (point 7 fix).
    Non-eligible tickers are excluded from the ranking pool entirely — they
    get no percentile (None downstream), not a 0/bottom-of-market rank."""
    eligible_features = {sym: f for sym, f in features.items() if eligible_by_symbol.get(sym)}
    return percentile_ranks(eligible_features, field)


# ─────────────────────────────────────────────
# V1.1: Structural RS / Trend Strength primitives
# ─────────────────────────────────────────────

def renormalized_weighted_sum(values, weights):
    """Local copy of build_dashboard_states.renormalized_weighted_sum (point
    48 there): only keys present in both values/weights with a non-None
    value contribute, their weights renormalized to sum to 1. Duplicated
    rather than cross-imported so the Feature Engine has no dependency on
    the States Engine that consumes its output (build_dashboard_states.py
    already imports from build_narratives.py; the reverse would invert the
    pipeline's data-flow direction)."""
    usable = {k: v for k, v in values.items() if v is not None and k in weights}
    if not usable:
        return None
    total_w = sum(weights[k] for k in usable)
    if total_w <= 0:
        return None
    return sum(values[k] * weights[k] for k in usable) / total_w


def clamp_0_100(v):
    if v is None:
        return None
    return round(max(0.0, min(100.0, v)), 1)


def compute_recent_leader_bootstrap(close_df, eligible_by_symbol, structural_weights, trend_weights,
                                     slope_lookback, persistence_lookback, memory_sessions, leader_entry_cfg):
    """Vectorized historical reconstruction (V1.1 point 10/13): rather than
    waiting `memory_sessions` real trading days for the daily-history
    hysteresis trail to accumulate, replay the SAME structural_rs /
    trend_strength formulas across the last `memory_sessions` rows of the
    already-fetched 260-session price history to determine whether each
    ticker was a Leader (per leader_entry thresholds) at ANY point in that
    window. Uses TODAY's eligible universe for the whole window (no
    historical Universe-Filter/market-cap history exists) — an accepted
    approximation. Returns {symbol: bool}; callers mark these
    bootstrap_recent_leader=true, to be phased out once real daily-history
    hysteresis has accumulated `memory_sessions` genuine snapshots (see
    build_dashboard_states.py)."""
    eligible_cols = [s for s in close_df.columns if eligible_by_symbol.get(s)]
    if not eligible_cols:
        return {}
    close_e = close_df[eligible_cols]

    ret_1m = close_e.pct_change(21) * 100
    ret_3m = close_e.pct_change(63) * 100
    ret_6m = close_e.pct_change(126) * 100
    ret_12m = close_e.pct_change(252) * 100

    pct_1m = ret_1m.rank(pct=True, axis=1) * 100
    pct_3m = ret_3m.rank(pct=True, axis=1) * 100
    pct_6m = ret_6m.rank(pct=True, axis=1) * 100
    pct_12m = ret_12m.rank(pct=True, axis=1) * 100

    sw = structural_weights
    sw_sum = (pct_1m.notna() * sw["rs_1m"] + pct_3m.notna() * sw["rs_3m"] +
              pct_6m.notna() * sw["rs_6m"] + pct_12m.notna() * sw["rs_12m"])
    structural_rs_hist = (pct_1m.fillna(0) * sw["rs_1m"] + pct_3m.fillna(0) * sw["rs_3m"] +
                           pct_6m.fillna(0) * sw["rs_6m"] + pct_12m.fillna(0) * sw["rs_12m"]) \
        / sw_sum.replace(0, np.nan)

    sma50_hist = close_e.rolling(50).mean()
    slope_pct_hist = (sma50_hist / sma50_hist.shift(slope_lookback) - 1) * 100
    slope_percentile_hist = slope_pct_hist.rank(pct=True, axis=1) * 100
    persistence_hist = (close_e > sma50_hist).rolling(persistence_lookback).mean() * 100

    tw = trend_weights
    tw_sum = (slope_percentile_hist.notna() * tw["slope_percentile"] + persistence_hist.notna() * tw["persistence"])
    trend_strength_hist = (slope_percentile_hist.fillna(0) * tw["slope_percentile"] +
                            persistence_hist.fillna(0) * tw["persistence"]) / tw_sum.replace(0, np.nan)

    leader_hist = (structural_rs_hist >= leader_entry_cfg["structural_rs_min"]) & \
                  (trend_strength_hist >= leader_entry_cfg["trend_strength_min"])

    window = leader_hist.iloc[-memory_sessions:]
    was_leader = window.any(axis=0)
    return {sym: bool(was_leader.get(sym, False)) for sym in eligible_cols}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Full-Market Feature Engine")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--taxonomy-dir", default="data/taxonomy")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--max-enrich-calls", type=int, default=4000,
                         help="Safety cap on per-ticker market-cap/SIC calls per run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_dir = Path(args.taxonomy_dir)

    print("=" * 60)
    print("🚀 YOLO Dashboard — Full-Market Feature Engine")
    print(f"   Zeit: {datetime.now().isoformat()}")
    print("=" * 60)

    cfg = load_config(args.config)
    u_cfg = cfg["universe"]
    cache_cfg = cfg["market_history_cache"]
    trend_cfg = cfg["trend_strength_v1_1"]
    # V6 point 9: RSP (S&P 500 Equal Weight ETF) is the new sole canonical
    # Narrative-Benchmark (replaces SPY). Folded into the SAME grouped-daily
    # walk as the stock universe (zero extra API calls) and persisted into
    # the shared price-history cache despite being an ETF -- exact same
    # pattern build_narratives.py already used for SPY -- then popped back
    # out of `features` below so RSP can NEVER reach eligibility, RS
    # percentiles, or market_features.json's tickers block.
    rsp_cfg = cfg["rsp_benchmark"]
    RSP_TICKER = rsp_cfg["ticker"]

    types_ref = load_types_reference(taxonomy_dir / "market_reference_types.json")
    type_universe = type_eligible_universe(types_ref, set(u_cfg["excluded_types"]))
    print(f"\n📋 {len(type_universe)} Ticker nach Asset-Type-Filter (ETF/ETN/FUND/... ausgeschlossen)")

    close_df, high_df, low_df, trading_days = fetch_grouped_history_full_market_cached(
        type_universe | {RSP_TICKER}, cache_cfg["path"], target_days=cache_cfg["target_trading_days"],
        backfill_tickers={RSP_TICKER})
    if len(trading_days) < 21:
        print("FATAL: Zu wenige Handelstage geladen, breche ab.", file=sys.stderr)
        sys.exit(1)

    daily_ret = close_df.pct_change() * 100

    features = calc_ticker_features(
        close_df, high_df, low_df, u_cfg["adr_lookback_sessions"],
        sma50_slope_lookback=trend_cfg["sma50_slope_lookback_sessions"],
        sma50_persistence_lookback=trend_cfg["sma50_persistence_lookback_sessions"])
    # RSP is a benchmark, never a stock -- pop it back out immediately so it
    # never flows into ADR candidates, enrichment, eligibility, RS/Thrust
    # percentiles, or output_tickers (point 9: "RSP darf NIE zum normalen
    # eligiblen Stock Universe gehoeren").
    features.pop(RSP_TICKER, None)
    print(f"  ✅ Preis-/ADR-/EMA-/ATR-Features fuer {len(features)}/{len(type_universe)} Ticker berechnet "
          f"({RSP_TICKER} als Benchmark aus dem Price-Cache mitgefuehrt, aus Features/Universe entfernt)")

    # ADR-eligible candidates (cheap, price-based) -> only these get the
    # expensive per-ticker market-cap/SIC lookup.
    adr_candidates = [sym for sym, f in features.items()
                       if f["adr20"] is not None and f["adr20"] > u_cfg["adr_minimum_pct"]]
    print(f"  → {len(adr_candidates)} Ticker erfuellen ADR{u_cfg['adr_lookback_sessions']} > {u_cfg['adr_minimum_pct']}% "
          f"(diese werden fuer Market-Cap/SIC angereichert, nicht der gesamte Markt)")

    enrich_cache_path = taxonomy_dir / "market_reference_cache.json"
    enrich_cache = ref.enrich_candidates(adr_candidates, enrich_cache_path, max_calls=args.max_enrich_calls)

    # market_cap resolved for every ticker with features (needed by
    # compute_eligible_universe below; falls back to shares*close when the
    # overview endpoint didn't return market_cap directly).
    market_cap_by_symbol = {}
    for sym, f in features.items():
        overview = enrich_cache.get(sym, {})
        shares = overview.get("share_class_shares_outstanding") or overview.get("weighted_shares_outstanding")
        market_cap = overview.get("market_cap")
        if market_cap is None and shares:
            market_cap = round(shares * f["close"], 2)
        market_cap_by_symbol[sym] = market_cap

    # V6 point 5-6: deterministic Healthcare/Biotech universe exclusion, MUST
    # run BEFORE eligible=true / stock RS-Thrust percentiles (spec pipeline
    # order: Asset-Type -> ADR20 -> Reference-Enrichment -> Healthcare/
    # Biotech-Exclusion -> Market-Cap -> eligible=true -> percentiles).
    sic_by_symbol = {sym: enrich_cache.get(sym, {}) for sym in features}
    healthcare_excluded_by_symbol = compute_healthcare_excluded_universe(features, sic_by_symbol, u_cfg)
    n_healthcare_excluded = sum(1 for excluded, _ in healthcare_excluded_by_symbol.values() if excluded)
    print(f"  ✅ {n_healthcare_excluded}/{len(features)} Ticker als Healthcare/Biotech ausgeschlossen "
          f"(deterministisch, SIC-Code-first — siehe classify_healthcare_biotech)")

    n_eligible_before_healthcare = sum(
        1 for sym in features
        if compute_eligibility(features[sym]["adr20"], market_cap_by_symbol[sym], u_cfg, healthcare_excluded=False))

    # Eligibility MUST be resolved BEFORE any percentile ranking (point 7
    # fix) — see compute_eligible_universe()/eligible_percentile_ranks().
    eligible_by_symbol = compute_eligible_universe(
        features, market_cap_by_symbol, u_cfg,
        healthcare_excluded_by_symbol={sym: v[0] for sym, v in healthcare_excluded_by_symbol.items()})
    n_eligible = sum(1 for v in eligible_by_symbol.values() if v)
    print(f"  ✅ {n_eligible}/{len(features)} Ticker eligible (ADR{u_cfg['adr_lookback_sessions']} > "
          f"{u_cfg['adr_minimum_pct']}% UND Market Cap >= ${u_cfg['market_cap_minimum_usd']:,.0f} UND "
          f"NICHT Healthcare/Biotech) — RS-/Thrust-Perzentile werden NUR gegen dieses Subset berechnet "
          f"(vor Healthcare/Biotech-Filter waeren es {n_eligible_before_healthcare} gewesen)")

    # RS percentiles across the eligible universe ONLY — this is the
    # "Full-Market RS" the Leadership score (and everything downstream:
    # Discovery Candidates, Market Regime Momentum, Opportunities) relies
    # on. Non-eligible tickers get rs_percentile_* = None: not ranked, never
    # silently treated as 0 (would misrepresent them as market-bottom).
    pct_field_by_horizon = {
        "1d": "d1_pct", "1w": "w1_pct", "1m": "m1_pct",
        # V1.1 point 3: 3M/6M/12M RS percentiles feed structural_rs below —
        # same eligible-only ranking rule as 1D/1W/1M (point 7 fix).
        "3m": "return_3m", "6m": "return_6m", "12m": "return_12m",
    }
    percentiles_by_horizon = {h: eligible_percentile_ranks(features, eligible_by_symbol, f)
                               for h, f in pct_field_by_horizon.items()}

    # V1.1 point 3-6: Structural RS (multi-timeframe RS, gates leadership)
    # and Trend Strength (SMA50 slope + persistence, trend QUALITY —
    # distance/extension deliberately excluded, that's Location not
    # structure). Both are 0-100 composites via the same renormalized-
    # weighted-sum primitive used throughout build_dashboard_states.py.
    slope_percentiles = eligible_percentile_ranks(features, eligible_by_symbol, "sma50_slope_20d_pct")
    structural_weights = cfg["structural_rs_v1_1"]["weights"]
    trend_weights = trend_cfg["weights"]

    # V1.1 point 10/13: bootstrap Recent Leader reconstruction from the same
    # 260-session history (no waiting on real daily-history hysteresis).
    recent_leader_bootstrap = compute_recent_leader_bootstrap(
        close_df, eligible_by_symbol, structural_weights, trend_weights,
        trend_cfg["sma50_slope_lookback_sessions"], trend_cfg["sma50_persistence_lookback_sessions"],
        cfg["opportunity_v1_1"]["recent_leader"]["memory_sessions"], cfg["opportunity_v1_1"]["leader_entry"])
    print(f"  ✅ Structural RS / Trend Strength / Recent-Leader-Bootstrap berechnet "
          f"({sum(recent_leader_bootstrap.values())} Ticker via Bootstrap als jüngste Leader erkannt)")

    # V6.1 point 16: Stock Thrust ("stock_thrust_rs") -- same formula shape
    # as build_narratives.py's Narrative Thrust (0.60*RS1W + 0.40*RS1M +
    # 0.10*(RS1W_today - RS1W_N_sessions_ago)), reusing
    # compute_narrative_thrust_rsp directly rather than duplicating the
    # formula (it's already a generic percentile-in/percentile-out function
    # despite its name). RS1W_N_sessions_ago is reconstructed straight from
    # close_df (the same 260-session price history everything else here
    # already uses): a 5-session return ending N sessions before the most
    # recent one, then percentile-ranked cross-sectionally over the SAME
    # eligible population as today's rs_percentile_1w -- never a second,
    # independent eligible-universe definition.
    thrust_delta_sessions = cfg["rsp_benchmark"]["thrust_delta_sessions"]
    thrust_weights_shared = cfg["rsp_benchmark"]["thrust_weights"]
    w1_window = HORIZONS["1w"]["window"]

    ret_1w_n_ago_by_symbol = {sym: calc_return_n_sessions_ago(close_df, sym, w1_window, thrust_delta_sessions)
                               for sym in features}
    rs_1w_n_ago_by_symbol = eligible_percentile_ranks(
        {sym: {"_r": v} for sym, v in ret_1w_n_ago_by_symbol.items() if v is not None},
        eligible_by_symbol, "_r")
    stock_thrust_rs_by_symbol = {
        sym: compute_narrative_thrust_rsp(
            percentiles_by_horizon["1w"].get(sym), percentiles_by_horizon["1m"].get(sym),
            rs_1w_n_ago_by_symbol.get(sym), thrust_weights_shared)
        for sym in features
    }

    thrust_by_horizon = {}
    for h, hcfg in HORIZONS.items():
        # Raw Thrust value computed for every ticker with features (used for
        # display/discovery even outside the eligible set); only the
        # PERCENTILE ranking below is restricted to the eligible universe.
        thrust_by_horizon[h] = {
            sym: calc_thrust(daily_ret, sym, hcfg["thrust_short"], hcfg["thrust_long"])
            for sym in features
        }

    # Thrust percentiles: same eligible-only ranking rule as RS (point 7).
    thrust_percentiles = {}
    for h in HORIZONS:
        thrust_features = {sym: {"_t": v} for sym, v in thrust_by_horizon[h].items() if v is not None}
        thrust_percentiles[h] = eligible_percentile_ranks(thrust_features, eligible_by_symbol, "_t")

    rs_pct = cfg["discovery"]["rs_candidate_percentile"]
    thrust_pct = cfg["discovery"]["thrust_candidate_percentile"]

    output_tickers = {}
    for sym, f in features.items():
        eligible = eligible_by_symbol[sym]
        market_cap = market_cap_by_symbol[sym]
        overview = enrich_cache.get(sym, {})

        healthcare_excluded, healthcare_reason = healthcare_excluded_by_symbol.get(sym, (False, None))

        rs_1w = percentiles_by_horizon["1w"].get(sym)
        rs_1m = percentiles_by_horizon["1m"].get(sym)
        rs_3m = percentiles_by_horizon["3m"].get(sym)
        rs_6m = percentiles_by_horizon["6m"].get(sym)
        rs_12m = percentiles_by_horizon["12m"].get(sym)
        thrust_1d_pct = thrust_percentiles["1d"].get(sym)
        thrust_1w_pct = thrust_percentiles["1w"].get(sym)
        thrust_1m_pct = thrust_percentiles["1m"].get(sym)

        structural_rs = clamp_0_100(renormalized_weighted_sum(
            {"rs_1m": rs_1m, "rs_3m": rs_3m, "rs_6m": rs_6m, "rs_12m": rs_12m},
            structural_weights))
        trend_strength = clamp_0_100(renormalized_weighted_sum(
            {"slope_percentile": slope_percentiles.get(sym), "persistence": f.get("pct_sessions_above_sma50_20d")},
            trend_weights))

        discovery_candidate = bool(
            (rs_1w is not None and rs_1w >= rs_pct) or
            (rs_1m is not None and rs_1m >= rs_pct) or
            (thrust_1d_pct is not None and thrust_1d_pct >= thrust_pct) or
            (thrust_1w_pct is not None and thrust_1w_pct >= thrust_pct)
        )

        output_tickers[sym] = {
            **f,
            "market_cap": market_cap,
            "eligible": eligible,
            "rs_percentile_1d": percentiles_by_horizon["1d"].get(sym),
            "rs_percentile_1w": rs_1w,
            "rs_percentile_1m": rs_1m,
            "rs_percentile_3m": rs_3m,
            "rs_percentile_6m": rs_6m,
            "rs_percentile_12m": rs_12m,
            "structural_rs": structural_rs,
            "sma50_slope_percentile": slope_percentiles.get(sym),
            "trend_strength": trend_strength,
            "bootstrap_recent_leader": recent_leader_bootstrap.get(sym, False),
            "thrust_1d": thrust_by_horizon["1d"].get(sym),
            "thrust_1w": thrust_by_horizon["1w"].get(sym),
            "thrust_1m": thrust_by_horizon["1m"].get(sym),
            "thrust_percentile_1d": thrust_1d_pct,
            "thrust_percentile_1w": thrust_1w_pct,
            "thrust_percentile_1m": thrust_1m_pct,
            "sic_code": overview.get("sic_code"),
            "sic_description": overview.get("sic_description"),
            # V6.1 point 18/19: exposed additively so both
            # classify_healthcare_biotech's SIC-present company_description
            # backstop (this file) and build_narratives.py's
            # find_healthcare_biotech_leaks sanity gate can audit it without
            # a second overview fetch.
            "company_description": overview.get("description"),
            "healthcare_biotech_excluded": healthcare_excluded,
            "healthcare_exclusion_reason": healthcare_reason,
            "stock_thrust_rs": stock_thrust_rs_by_symbol.get(sym),
            "discovery_candidate": discovery_candidate and eligible,
        }

    output = {
        "meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Massive (grouped daily + reference/tickers)",
            "trading_days_used": len(trading_days),
            "date_range": [trading_days[0], trading_days[-1]] if trading_days else None,
            "type_universe_size": len(type_universe),
            "features_computed": len(features),
            "adr_candidates_enriched": len(adr_candidates),
            "healthcare_biotech_excluded_count": n_healthcare_excluded,
            "eligible_count_before_healthcare_exclusion": n_eligible_before_healthcare,
            "eligible_count_after_healthcare_exclusion": n_eligible,
            "eligible_count": n_eligible,
            "discovery_candidate_count": sum(1 for t in output_tickers.values() if t["discovery_candidate"]),
            "structural_rs_computed_count": sum(1 for t in output_tickers.values() if t["structural_rs"] is not None),
            "trend_strength_computed_count": sum(1 for t in output_tickers.values() if t["trend_strength"] is not None),
            "bootstrap_recent_leader_count": sum(recent_leader_bootstrap.values()),
            "stock_thrust_rs_computed_count": sum(1 for t in output_tickers.values() if t["stock_thrust_rs"] is not None),
        },
        "tickers": output_tickers,
    }

    out_path = out_dir / "market_features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n✅ Market Features geschrieben → {out_path} ({size_kb:.0f} KB)")
    print(f"   Universe (Type-Filter): {len(type_universe)} | Features: {len(features)} | "
          f"Eligible: {n_eligible} | Discovery Candidates: {output['meta']['discovery_candidate_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
