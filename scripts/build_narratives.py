#!/usr/bin/env python3
"""
YOLO Dashboard — Narratives/Baskets Builder
Fetches daily grouped OHLC from the Massive API for the narrative-taxonomy
ticker set, computes per-ticker performance + basket-level Strength/Thrust/
Leadership/Breadth scores for the 1D/1W/1M horizons, and writes
data/narratives.json.

Score definitions (Strength/Thrust/Breadth unchanged since the first pass —
see point 24 of the Full-Market Narrative Engine spec: existing formulas are
not changed without a documented reason):
  Strength   — basket's cumulative return over the horizon (median of member
               daily returns compounded across the window).
  Thrust     — EMA(short) - EMA(long) of the basket's daily median-return
               series; measures fresh acceleration rather than raw level.
  Leadership — share of basket members ranking in the top quintile for that
               horizon, saturating once ~5 members qualify so many leaders
               score higher than one outlier. CHANGED: the percentile rank
               now comes from data/market_features.json, i.e. the Full-
               Market RS (rank against the whole eligible US stock universe)
               instead of the curated narrative-taxonomy ticker set. This is
               the one formula change in this file, and it is the one the
               spec explicitly asked for.
  Breadth    — % of members positive, median member return, % of members
               beating a horizon-scaled "significant move" threshold —
               computed over ALL members currently carried for the basket
               (not just winners, no survivorship/strength pre-selection).

Taxonomy source: data/taxonomy/narratives.json (Source of Truth, migrated
from the legacy data/narratives_map.json — see scripts/migrate_taxonomy.py).
Falls back to the legacy file with a warning if the new one is not present,
so this script stays independently runnable (e.g. for local testing).

Price history for Strength/Thrust is still fetched here directly (own
grouped-daily walk-back over just the taxonomy universe), NOT read from
market_features.json: the basket aggregates need the full daily-return
*time series* across the window, which market_features.json deliberately
does not persist per ticker (that would multiply its size by the lookback
window for no other consumer — see the Full-Market Feature Engine's
payload-size reasoning in the technical report). Re-fetching a ~55-trading-
day window for ~100-300 taxonomy tickers is a small, bounded cost, separate
from the market-wide walk in build_market_features.py.

Benchmark RS history (dashboard "Benchmark" subsection under Narratives):
RSP (S&P 500 Equal Weight ETF) is folded into the same shared price-history
cache walk as the narrative universe (zero extra API calls), then popped
back out of ticker_metrics so it never appears as a narrative member or
pollutes the percentile-fallback pool. See compute_narrative_rs_history()
for the basket-vs-RSP cumulative-return calculation and data/narratives.json's
"rs_history" key for the output shape (one shared "dates" array + per-
narrative value lists). data/narratives.json's "benchmark_rsp" key carries
the raw RSP close/date series itself (the SAME prices[RSP_TICKER] column
used below for Narrative Strength) for the dashboard's dedicated RSP
Benchmark chart — see build_benchmark_rsp_series().

V6 (Navigation/Universe-Filter/Screener/Narrative-Strength synthesis) —
new RSP-based Narrative Strength/Thrust, replacing SPY as the benchmark:
- RSP replaces SPY everywhere in this file (config rsp_benchmark.ticker).
  RSP is a benchmark ETF, NEVER eligible, NEVER a narrative member, NEVER
  in the stock RS percentile pool.
- Narrative Strength ("Jeff-inspired Relative Strength gegen RSP", a
  candidate v1 formula -- explicitly NOT claimed to be Jeff Sun's exact
  unpublished formula): equal-weight MEAN daily return of a narrative's
  currently-active members (compute_narrative_equal_weight_return_series)
  compounded into a synthetic index starting at 100
  (build_synthetic_narrative_index), divided by the canonical RSP close on
  the same real trading sessions (compute_relative_strength_line), then
  percentile-ranked against its OWN trailing 1W/5-, 1M/20-, 3M/63-, 6M/126-
  session window (percentile_rank_of_current, reusing the exact same
  sort-position convention as percentile_ranks() above) -> strength_rsp
  {1w,1m,3m,6m}, fully separate values, never averaged/composited.
- Narrative Thrust ("Jeff-inspired Thrust candidate v1"):
  0.60*Strength_1W(today) + 0.40*Strength_1M(today) +
  0.10*(Strength_1W(today) - Strength_1W(3 real trading sessions ago)) ->
  thrust_rsp. NOT clamped to 0-100. thrust_rsp is None unless all three
  inputs are present -- no alternative weighting is ever substituted for a
  missing input.
- The OLD basket-median-return Strength (scores[h]["strength"]) and OLD
  EMA-difference Thrust (scores[h]["thrust"]) are DEPRECATED for the
  VISIBLE dashboard as of V6 but are intentionally left completely
  unchanged/uncomputed-different here: narrative_structural_score,
  momentum_modifier and thrust_percentile_1w still consume them internally
  exactly as before (V1.1 methodology, untouched). The frontend must
  display strength_rsp/thrust_rsp as the new headline Strength/Thrust and
  must never surface scores[h]["strength"]/["thrust"] as if they were the
  new metric -- see index.html's Narrative card rendering (task point 29).
- Point 28 (3-session Thrust delta, must use real trading sessions, not
  calendar days): implemented via variant "deterministically recompute
  historical Strength_1W from the already-available OHLC/price history"
  (NOT the data/history/daily/ per-day snapshot architecture) --
  strength_percentile_at() slices the SAME relative-strength line 3
  positions further back on its own (real-session-only, gap-free by
  construction — see build_synthetic_narrative_index) index. No lookahead:
  every window used only ever looks backward from its reference session.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd
import numpy as np

MASSIVE_BASE = "https://api.massive.com"

HORIZONS = {
    "1d": {"window": 1, "thrust_short": 2, "thrust_long": 5, "sig_threshold": 2.0},
    "1w": {"window": 5, "thrust_short": 5, "thrust_long": 15, "sig_threshold": 5.0},
    "1m": {"window": 21, "thrust_short": 10, "thrust_long": 25, "sig_threshold": 10.0},
    # V1.1 point 17: 3M/6M horizons for the Narrative Structural Score, added
    # to the SAME dict so calc_basket_scores() computes Strength/Thrust/
    # Leadership/Breadth for them via the identical, unchanged methodology
    # (no new formula — just two more horizons in the existing loop).
    "3m": {"window": 63, "thrust_short": 15, "thrust_long": 40, "sig_threshold": 15.0},
    "6m": {"window": 126, "thrust_short": 25, "thrust_long": 60, "sig_threshold": 20.0},
}

TRADING_DAYS_NEEDED = 55   # 50 trading days (~10 weeks) for the Benchmark RS
                            # history chart + buffer above the 21-day 1M window.
                            # Only used as the FALLBACK walk-back window when
                            # the shared V1.1 price-history cache (see
                            # load_shared_price_cache_frame) isn't available —
                            # 3M/6M metrics stay None in that fallback path,
                            # same graceful-degradation rule as every other
                            # lookback-dependent field in this pipeline.
MAX_CALENDAR_LOOKBACK = 90  # safety cap so weekends/holidays can't loop forever
RS_HISTORY_LOOKBACK_DAYS = 50  # ~10 trading weeks, matches the Benchmark chart
MIN_RESULTS_FOR_TRADING_DAY = 1000  # grouped-daily returns ~12k rows on a real session
SHARED_CACHE_MIN_TRADING_DAYS = 146  # 126 (6M window) + 20 (SMA50-slope buffer)


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


def load_taxonomy(path, legacy_path="data/narratives_map.json"):
    """Reads the structured Source of Truth (data/taxonomy/narratives.json:
    tickers as {SYMBOL: {role, assignment_priority, confidence, ...}}).
    Falls back to the legacy flat-list format (data/narratives_map.json:
    tickers as [SYMBOL, ...]) with a warning if the new file isn't present
    yet — legacy narratives get no membership_meta (assignment_priority
    lookups degrade to None, same graceful-degradation rule as everywhere
    else in this pipeline).

    `tickers` stays a plain sorted list (every existing caller iterates it
    as such); `membership_meta` is new and carries the per-ticker
    assignment_priority/role needed for the Full-Universe Primary/Secondary
    Narrative context (Opportunities linkage) without disturbing any
    existing consumer."""
    p = Path(path)
    if not p.exists():
        print(f"  ⚠ {path} nicht gefunden, falle zurueck auf Legacy-Taxonomie {legacy_path}", file=sys.stderr)
        p = Path(legacy_path)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    narratives = []
    for n in data["narratives"]:
        tickers = n["tickers"]
        if isinstance(tickers, dict):
            member_list = sorted(tickers.keys())
            membership_meta = {sym: {"assignment_priority": meta.get("assignment_priority"),
                                       "role": meta.get("role")}
                                for sym, meta in tickers.items()}
        else:
            member_list = list(tickers)
            membership_meta = {}
        narratives.append({
            "id": n["id"],
            "name": n["name"],
            "status": n.get("status", "active"),
            "tickers": member_list,
            "membership_meta": membership_meta,
        })
    universe = sorted({t for n in narratives for t in n["tickers"]})
    return narratives, universe


def load_market_features(path):
    """Full-Market feature set from build_market_features.py. Returns None
    (with a warning) if not present, so this script degrades gracefully to
    curated-universe percentiles instead of crashing — e.g. for local runs
    that only exercise the narrative builder on its own."""
    p = Path(path)
    if not p.exists():
        print(f"  ⚠ {path} nicht gefunden — Leadership/RS fallen zurueck auf das kuratierte "
              "Taxonomie-Universum statt Full-Market (Full-Market-Migration nicht aktiv fuer diesen Lauf)",
              file=sys.stderr)
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["tickers"]


def load_shared_price_cache_frame(cache_path, universe_set):
    """Read the persistent rolling-window price cache written by
    build_market_features.py (config market_history_cache, V1.1 point 2) as
    a close-price DataFrame restricted to `universe_set` — zero extra API
    calls, this script reuses the SAME cache instead of its own market-wide
    walk-back. Returns (close_df, trading_days) or (None, None) if the cache
    is absent or doesn't cover enough sessions for the new 3M/6M structural
    horizons (SHARED_CACHE_MIN_TRADING_DAYS); callers fall back to
    fetch_grouped_history()'s shorter, narrative-taxonomy-scoped walk in
    that case (same graceful-degradation pattern as load_market_features)."""
    p = Path(cache_path)
    if not p.exists():
        return None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    trading_days = cache.get("dates", [])
    if len(trading_days) < SHARED_CACHE_MIN_TRADING_DAYS:
        print(f"  ⚠ Geteilter Preis-Cache zu kurz ({len(trading_days)} < {SHARED_CACHE_MIN_TRADING_DAYS} "
              "Handelstage) — falle zurueck auf eigenen Walk-back (3M/6M-Felder bleiben in diesem Lauf None)",
              file=sys.stderr)
        return None, None
    tickers = cache.get("tickers", {})
    cols = {sym: tickers[sym]["close"] for sym in universe_set if sym in tickers}
    if not cols:
        return None, None
    close_df = pd.DataFrame(cols, index=trading_days)
    print(f"  ✅ Geteilter Preis-Cache geladen: {len(trading_days)} Handelstage, "
          f"{len(cols)}/{len(universe_set)} Ticker (0 zusaetzliche API-Calls)")
    return close_df, trading_days


def fetch_grouped_history(universe_set):
    """Walk backward day by day, collecting grouped-daily closes for tickers
    in our curated universe until we have enough trading days."""
    print(f"\n📊 Lade Grouped-Daily-OHLC (Ziel: {TRADING_DAYS_NEEDED} Handelstage)...")
    per_ticker = {t: {} for t in universe_set}
    trading_days = []
    day = datetime.now(timezone.utc).date()
    calendar_checked = 0

    while len(trading_days) < TRADING_DAYS_NEEDED and calendar_checked < MAX_CALENDAR_LOOKBACK:
        date_str = day.isoformat()
        data = massive_get(f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}")
        calendar_checked += 1
        day -= timedelta(days=1)

        if not data or data.get("resultsCount", 0) < MIN_RESULTS_FOR_TRADING_DAY:
            continue

        trading_days.append(date_str)
        for row in data.get("results", []):
            sym = row.get("T")
            if sym in per_ticker:
                per_ticker[sym][date_str] = row.get("c")
        print(f"  → {date_str}: {data['resultsCount']} Ticker (Handelstag {len(trading_days)}/{TRADING_DAYS_NEEDED})")

    trading_days.sort()  # oldest → newest
    print(f"  ✅ {len(trading_days)} Handelstage geladen ({calendar_checked} Kalendertage geprüft)")
    return per_ticker, trading_days


def build_price_frame(per_ticker, trading_days):
    """DataFrame: index=trading_days (asc), columns=tickers, values=close."""
    df = pd.DataFrame(index=trading_days, columns=sorted(per_ticker.keys()), dtype=float)
    for sym, series in per_ticker.items():
        for date_str, close in series.items():
            if date_str in df.index:
                df.at[date_str, sym] = close
    return df


def calc_ticker_metrics(prices):
    """prices: DataFrame (dates asc x tickers). Returns per-ticker dict."""
    daily_ret = prices.pct_change() * 100  # % daily returns per ticker

    out = {}
    for sym in prices.columns:
        s = prices[sym].dropna()
        if len(s) < 2:
            continue
        last = s.iloc[-1]

        def pct_ago(n):
            if len(s) > n:
                base = s.iloc[-1 - n]
                return round(float((last - base) / base * 100), 2) if base else None
            return None

        hist_1w = [round(float(x), 2) for x in s.iloc[-6:].tolist()]
        hist_1m = [round(float(x), 2) for x in s.iloc[-22:].tolist()]
        # V6.1 (Narrative Ranking & UI Bugfix Patch) point 14: absolute
        # change (current_close - previous_close) for the new Member-Detail-
        # Tabelle's "Veränderung absolut" column -- d1_pct already covers
        # "Veränderung %" (mathematically identical to the spec's own
        # (current/previous-1)*100, reused per point 14's explicit allowance).
        change_abs = round(float(last - s.iloc[-2]), 2) if len(s) > 1 else None

        out[sym] = {
            "symbol": sym,
            "price": round(float(last), 2),
            "change_abs": change_abs,
            "d1_pct": pct_ago(1),
            "w1_pct": pct_ago(5),
            "m1_pct": pct_ago(21),
            "return_3m": pct_ago(63),   # V1.1 point 17: feeds the 3M horizon in HORIZONS
            "return_6m": pct_ago(126),  # V1.1 point 17: feeds the 6M horizon in HORIZONS
            "hist_1w": hist_1w,
            "hist_1m": hist_1m,
        }
    return out, daily_ret


def percentile_ranks(ticker_metrics, field):
    """Return {symbol: percentile 0-100} across the tracked universe for a field."""
    vals = {sym: m[field] for sym, m in ticker_metrics.items() if m.get(field) is not None}
    if not vals:
        return {}
    ordered = sorted(vals.items(), key=lambda kv: kv[1])
    n = len(ordered)
    ranks = {}
    for i, (sym, _) in enumerate(ordered):
        ranks[sym] = round((i + 1) / n * 100, 1)
    return ranks


# ─────────────────────────────────────────────
# V1.1: Narrative Structural Score primitives
# ─────────────────────────────────────────────

def renormalized_weighted_sum(values, weights):
    """Local copy of build_dashboard_states.renormalized_weighted_sum /
    build_market_features.renormalized_weighted_sum (point 48 there): only
    keys present in both values/weights with a non-None value contribute,
    their weights renormalized to sum to 1. Duplicated (not cross-imported)
    to avoid a circular import — build_market_features.py already imports
    FROM this module (percentile_ranks/HORIZONS)."""
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


def calc_trend_participation(members, market_features):
    """Trend Participation (V1.1 point 20): share of narrative members whose
    price sits above SMA50, whose SMA50 is rising, and both simultaneously —
    a STRUCTURAL breadth measure (independent of any single leader's short-
    term push). Members without market_features data are excluded from the
    denominator, not counted as failing — same graceful-degradation rule as
    everywhere else in this pipeline. Returns
    (pct_above_sma50, pct_rising_sma50, pct_above_rising_sma50), any/all
    None if no member has usable data."""
    mf = market_features or {}
    above_flags, rising_flags, both_flags = [], [], []
    for m in members:
        rec = mf.get(m)
        if not rec:
            continue
        dist = rec.get("sma50_distance_pct")
        slope = rec.get("sma50_slope_20d_pct")
        if dist is None or slope is None:
            continue
        is_above = dist > 0
        is_rising = slope > 0
        above_flags.append(is_above)
        rising_flags.append(is_rising)
        both_flags.append(is_above and is_rising)
    if not above_flags:
        return None, None, None
    n = len(above_flags)
    pct_above = round(sum(above_flags) / n * 100, 1)
    pct_rising = round(sum(rising_flags) / n * 100, 1)
    pct_above_rising = round(sum(both_flags) / n * 100, 1)
    return pct_above, pct_rising, pct_above_rising


def calc_structural_leadership_pct(members, market_features, threshold):
    """% of narrative members whose structural_rs (build_market_features.py,
    V1.1 point 3) is at/above `threshold` — structural (multi-timeframe RS)
    leadership share, replacing the old short-term-RS-based Leadership
    formula as an input to the Narrative Structural Score. Members without
    a computed structural_rs are excluded from the denominator."""
    mf = market_features or {}
    vals = [mf[m]["structural_rs"] for m in members
            if mf.get(m) and mf[m].get("structural_rs") is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v >= threshold) / len(vals) * 100, 1)


def calc_momentum_modifier(thrust_1w, thrust_percentile_1w, structural_score, modifier_cfg):
    """ACCELERATING/COOLING/None (V1.1 point 22) — a separate, non-gating
    annotation on top of the structural score, NOT a component of it and
    NOT the Lifecycle stage: Lifecycle (build_dashboard_states.py) tracks
    the multi-day/week structural trajectory, this flags short-term (1W)
    Thrust piling on top of (accelerating) or fading under (cooling) an
    already-decided structural read."""
    acc = modifier_cfg["accelerating"]
    cool = modifier_cfg["cooling"]
    if (thrust_1w is not None and thrust_percentile_1w is not None
            and (thrust_1w > 0) == acc["thrust_1w_positive_required"]
            and thrust_percentile_1w >= acc["thrust_percentile_1w_min"]):
        return "ACCELERATING"
    if (structural_score is not None and thrust_1w is not None
            and structural_score >= cool["structural_score_min"]
            and (thrust_1w < 0) == cool["thrust_1w_negative_required"]):
        return "COOLING"
    return None


def basket_daily_return_series(daily_ret, members):
    """Median daily % return of basket members, per trading day."""
    cols = [m for m in members if m in daily_ret.columns]
    if not cols:
        return pd.Series(dtype=float)
    return daily_ret[cols].median(axis=1, skipna=True)


def compute_narrative_rs_history(narratives, relative_strength_by_id, trading_days,
                                  lookback_days=RS_HISTORY_LOOKBACK_DAYS, benchmark_ticker="RSP"):
    """RVOL/Screener/Benchmark/Futures Patch point 10: adapts this backend
    structure to the CURRENT Narrative Strength/RSP methodology instead of
    reactivating the old median-basket-return-minus-benchmark-return method.
    `relative_strength_by_id` is the SAME {narrative_id: pd.Series} the
    headline narrative_rs percentiles above are built from (narrative_index_t
    / rsp_close_t, see compute_relative_strength_line) — zero second
    calculation model, just a different slice of the same lines for the
    multi-narrative-pill comparison chart (the restored old presentation,
    spec point 9).

    chart_relative_pct_t = (relative_strength_t / relative_strength_start - 1) * 100,
    where relative_strength_start is the FIRST valid value within this
    narrative's own visible lookback window — every line therefore starts at
    exactly 0% (point 10), and the dashed 0% baseline the chart draws
    represents RSP itself: above 0 = narrative outperforming RSP since chart
    start, below 0 = underperforming.

    A narrative needs at least 10 valid points within the window to be
    included (same threshold the old function used) — returns None (with a
    warning) if NO narrative qualifies, so the frontend can show its
    existing empty-state message rather than crash."""
    if not trading_days:
        return None
    window_dates = trading_days[-lookback_days:]

    series_by_narrative = {}
    for n in narratives:
        rel = relative_strength_by_id.get(n["id"])
        if rel is None:
            continue
        windowed = rel.reindex(window_dates)
        valid = windowed.dropna()
        if len(valid) < 10:
            continue
        start = valid.iloc[0]
        if not start:
            continue
        chart_pct = (windowed / start - 1.0) * 100
        series_by_narrative[n["id"]] = [None if pd.isna(v) else round(float(v), 2) for v in chart_pct.tolist()]

    if not series_by_narrative:
        print(f"  ⚠ Keine Narrative mit ausreichender {benchmark_ticker}-relativer Historie — Benchmark-RS-Historie übersprungen",
              file=sys.stderr)
        return None

    dates_fmt = [datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.") for d in window_dates]
    return {
        "benchmark": benchmark_ticker,
        "lookback_trading_days": len(window_dates),
        "dates": dates_fmt,
        "narratives": series_by_narrative,
    }


# ─────────────────────────────────────────────
# V6: RSP-based Narrative Strength/Thrust ("Jeff-inspired" candidate v1)
# ─────────────────────────────────────────────

def compute_narrative_equal_weight_return_series(daily_ret, members):
    """Equal-weight MEAN daily % return of a narrative's currently-active
    members (point 10) — explicitly MEAN, not the median used by the legacy
    basket_daily_return_series/old Strength. `.mean(axis=1, skipna=True)`
    already implements the exact missing-data rule the spec requires: a
    member missing its return on a given day is excluded from just that
    day's mean (skipna); a day where ALL members are missing yields NaN
    (pandas mean-of-all-NaN), which build_synthetic_narrative_index() below
    then DROPS entirely rather than fabricating/forward-filling a value."""
    cols = [m for m in members if m in daily_ret.columns]
    if not cols:
        return pd.Series(dtype=float)
    return daily_ret[cols].mean(axis=1, skipna=True)


def build_synthetic_narrative_index(return_series):
    """Synthetic equal-weight narrative index, starting at 100 (point 10):
    narrative_close_t = narrative_close_(t-1) * (1 + narrative_return_t).
    Days with no valid narrative return (NaN — see
    compute_narrative_equal_weight_return_series) are DROPPED from the
    index entirely, never inserted as an artificial flat/forward-filled
    point — the returned Series' own index is therefore the narrative's
    own real, valid-return trading sessions only (a subsequence of the
    full market calendar, gap-free by construction)."""
    valid = return_series.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    return (1 + valid / 100.0).cumprod() * 100.0


def compute_relative_strength_line(narrative_index, benchmark_close):
    """relative_strength_t = narrative_close_t / benchmark_close_t (point 11
    step 2), aligned on real, shared trading sessions only: benchmark_close
    is reindexed onto narrative_index's own (already valid-only) date axis,
    and any date where the benchmark itself has no price is additionally
    dropped — never fabricated on either side."""
    if narrative_index.empty:
        return pd.Series(dtype=float)
    bm = benchmark_close.reindex(narrative_index.index)
    aligned = pd.DataFrame({"narrative": narrative_index, "benchmark": bm}).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["narrative"] / aligned["benchmark"]


def relative_performance_at(relative_strength, window, sessions_ago=0):
    """V6.1 (Narrative Ranking & UI Bugfix Patch) point 6.3/7: the RAW
    relative-performance return over `window` real trading sessions, as of
    `sessions_ago` sessions before the most recent one available in
    `relative_strength` (point 9's historical-RS1W-for-Thrust reconstruction
    reuses this with sessions_ago>0). Mathematically identical to
    (1+narrative_horizon_return)/(1+rsp_horizon_return)-1 since
    relative_strength_t = narrative_index_t / rsp_close_t by construction
    (point 6.3 explicitly permits this equivalent ratio-based formulation —
    no second calculation model). Returns the raw ratio-based return, NOT a
    percentile — cross_sectional_percentile_ranks() below turns a whole
    snapshot of these into the actual Narrative RS. None if there isn't a
    full `window`-session gap before the reference session (graceful
    degradation, never a partial/padded window)."""
    valid = relative_strength.dropna()
    n = len(valid)
    end = n - sessions_ago
    if end - 1 - window < 0:
        return None
    current = valid.iloc[end - 1]
    base = valid.iloc[end - 1 - window]
    if not base:
        return None
    return float(current / base - 1.0)


def cross_sectional_percentile_ranks(values):
    """V6.1 point 7: cross-sectional percentile rank (0-100) of each id's
    value against ALL OTHER ids with a value in this SAME snapshot — this
    is what actually makes Narrative RS answer 'how does this narrative's
    relative performance compare to every OTHER active narrative right now',
    replacing V6's self-window percentile_rank_of_current (which answered
    the wrong question: 'where does today sit within THIS narrative's own
    recent history', collapsing to a handful of discrete values like
    20/40/60/80/100 whenever the window was short, e.g. 5 sessions for 1W).
    Point 7's explicit Tie-Handling requirement: exactly-equal values get
    the EXACT SAME rank via the average-rank method, never an order-
    dependent stable-sort position (unlike percentile_ranks() above, which
    intentionally keeps that simpler tie-break for its own unrelated
    ticker-vs-ticker/narrative-vs-narrative-Strength callers) — equivalent
    to pandas.Series(values).rank(method='average', pct=True) * 100."""
    usable = {k: v for k, v in values.items() if v is not None}
    if not usable:
        return {}
    keys = list(usable.keys())
    ranks = pd.Series([usable[k] for k in keys]).rank(method="average", pct=True) * 100
    return {k: round(float(r), 1) for k, r in zip(keys, ranks)}


def compute_narrative_rs(relative_strength_by_id, windows_sessions, sessions_ago=0):
    """V6.1 point 7-8: {"1w": {id: rs}, "1m": {...}, "3m": {...}, "6m": {...}}
    PLUS the raw relative_performance dict per window, cross-sectional across
    ALL ids with a computable relative_strength line at `sessions_ago`. Each
    window fully separate (point 7: 'keine Composite Strength') — no
    averaging/weighting across 1W/1M/3M/6M."""
    narrative_rs, relative_performance = {}, {}
    for label, window in windows_sessions.items():
        raw = {nid: relative_performance_at(rel, window, sessions_ago)
               for nid, rel in relative_strength_by_id.items()}
        relative_performance[label] = raw
        narrative_rs[label] = cross_sectional_percentile_ranks(raw)
    return narrative_rs, relative_performance


def compute_narrative_thrust_rsp(strength_1w_today, strength_1m_today, strength_1w_n_sessions_ago, thrust_weights):
    """Thrust ('Jeff-inspired Thrust candidate v1', point 12):
      0.60*Strength_1W(today) + 0.40*Strength_1M(today)
      + 0.10*(Strength_1W(today) - Strength_1W(N sessions ago))
    Deliberately NOT clamped to 0-100 (may legitimately exceed 100 or go
    negative). None unless ALL three inputs are present — no alternative
    weighting/renormalization is ever substituted for a missing input
    (unlike renormalized_weighted_sum elsewhere in this pipeline, which is
    intentionally NOT reused here). V6.1: the formula shape itself is
    UNCHANGED — only its inputs changed, from V6's self-window Strength
    percentiles to the new cross-sectional Narrative RS (compute_narrative_rs)
    / Stock RS (build_market_features.py's rs_percentile_1w/1m, which reuses
    this exact function for stock_thrust_rs -- same formula, two different
    percentile inputs, still ONE formula definition)."""
    if strength_1w_today is None or strength_1m_today is None or strength_1w_n_sessions_ago is None:
        return None
    w = thrust_weights
    acceleration = strength_1w_today - strength_1w_n_sessions_ago
    thrust = (w["strength_1w"] * strength_1w_today + w["strength_1m"] * strength_1m_today
              + w["delta_1w_acceleration"] * acceleration)
    return round(thrust, 2)


def build_benchmark_rsp_series(prices, rsp_ticker):
    """Raw RSP close/date series (point 29B/29C) for the dashboard's
    dedicated Benchmark card — the SAME prices[rsp_ticker] column used above
    for the Narrative Strength relative-strength line, so there is exactly
    ONE canonical RSP data source for both consumers (point 29B: 'keine
    zweite, unabhaengige Benchmark-Datenpipeline'). Time-sorted (prices is
    already indexed by ascending trading_days); NaN entries dropped (never
    passed to the frontend renderer). The frontend computes 1D/1W/1M/1Y
    cumulative returns client-side from this single payload — no server-side
    timeframe pre-slicing, so switching timeframe buttons needs zero extra
    API/data calls (point 29B's explicit requirement)."""
    if rsp_ticker not in prices.columns:
        return None
    s = prices[rsp_ticker].dropna()
    if s.empty:
        return None
    return {
        "ticker": rsp_ticker,
        "dates": list(s.index),
        "close": [round(float(v), 4) for v in s.tolist()],
    }


def find_healthcare_biotech_leaks(output_narratives, market_features, audit_keywords, min_active_members):
    """V6.1 (Narrative Ranking & UI Bugfix Patch) point 19: Healthcare/
    Biotech Sanity Gate — a DEFENSE-IN-DEPTH check run AFTER the build, never
    a second universe filter (audit_keywords is deliberately broader/looser
    than universe.healthcare_biotech_filter's precise exclusion keywords —
    point 19's explicit 'nicht als alleinigen Universe-Filter'). Flags an
    active narrative (>= `min_active_members` eligible members, the SAME
    threshold that already gates active/undersized elsewhere in this file,
    never a second hardcoded '5') if EITHER its own name reads as Healthcare/
    Biotech, OR more than half its eligible members individually do (via
    sic_description substring match ONLY -- see below) — catching both an
    unmistakably-named leak (point 18's real example: an active 'Biotech'
    narrative) and a content-concentrated one that happens to be named
    something neutral. Returns a list of violation dicts (empty if clean);
    the caller (main()) turns a non-empty result into sys.exit(1) with a full
    diagnostic dump, per point 19's explicit 'Build/Test fehlschlagen lassen'.

    The per-member signal deliberately checks ONLY sic_description (a
    controlled, standardized vocabulary), never company_description (free
    text) -- a first version that also matched company_description produced
    a real false positive here: 'Life Sciences Tools & Consumables'
    (AVTR/BRKR/NEO, all laboratory-instrument/testing-lab companies whose
    SIC -- 3826/8734 -- is deliberately NOT in the exclusion filter, see
    config's own comment) got flagged because their company_description
    mentions serving 'biopharma'/'healthcare'/'life sciences' CUSTOMERS,
    exactly the customer-vertical-mention false-positive point 6/18
    warns against. sic_description doesn't have that failure mode."""
    def matched_keywords(text):
        t = (text or "").lower()
        return [kw for kw in audit_keywords if kw in t]

    violations = []
    for row in output_narratives:
        if len(row["members"]) < min_active_members:
            continue
        name_hits = matched_keywords(row["name"])
        flagged_members = []
        for m in row["members"]:
            sym = m["symbol"]
            mf = (market_features or {}).get(sym, {})
            hits = matched_keywords(mf.get("sic_description"))
            if hits:
                flagged_members.append({
                    "symbol": sym, "sic_code": mf.get("sic_code"),
                    "sic_description": mf.get("sic_description"),
                    "company_description": mf.get("company_description"),
                    "matched_keywords": hits,
                })
        member_majority = len(row["members"]) > 0 and len(flagged_members) / len(row["members"]) > 0.5
        if name_hits or member_majority:
            violations.append({
                "narrative_id": row["id"],
                "narrative_name": row["name"],
                "eligible_member_count": len(row["members"]),
                "name_matched_keywords": name_hits,
                "flagged_member_count": len(flagged_members),
                "flagged_members": flagged_members,
            })
    return violations


def calc_basket_scores(members, ticker_metrics, daily_ret, pct_field_by_horizon, percentiles_by_horizon):
    scores = {}
    basket_ret_series = basket_daily_return_series(daily_ret, members)

    for h, cfg in HORIZONS.items():
        pct_field = pct_field_by_horizon[h]
        member_vals = [ticker_metrics[m][pct_field] for m in members
                       if m in ticker_metrics and ticker_metrics[m].get(pct_field) is not None]

        # Strength — compounded basket return over the window
        window_ret = basket_ret_series.iloc[-cfg["window"]:] / 100.0
        if len(window_ret) > 0:
            strength = round((float(np.prod(1 + window_ret.fillna(0))) - 1) * 100, 2)
        else:
            strength = None

        # Thrust — EMA(short) - EMA(long) of the basket daily-return series
        if len(basket_ret_series.dropna()) >= cfg["thrust_long"]:
            ema_short = basket_ret_series.ewm(span=cfg["thrust_short"], adjust=False).mean()
            ema_long = basket_ret_series.ewm(span=cfg["thrust_long"], adjust=False).mean()
            thrust = round(float(ema_short.iloc[-1] - ema_long.iloc[-1]), 2)
        else:
            thrust = None

        # Leadership — share of members in top quintile, saturating at ~5.
        # `percentiles_by_horizon[h]` is Full-Market RS when market_features.json
        # was available (see main()), else the curated-universe fallback.
        ranks = percentiles_by_horizon[h]
        top_quintile = [m for m in members if (ranks.get(m) or 0) >= 80]
        leadership = round(min(len(top_quintile) / min(len(members), 5), 1.0) * 100, 1)

        # Breadth — % positive, median, % beating a horizon-scaled significant-move threshold
        if member_vals:
            pct_positive = round(sum(1 for v in member_vals if v > 0) / len(member_vals) * 100, 1)
            median_pct = round(float(np.median(member_vals)), 2)
            pct_significant = round(sum(1 for v in member_vals if v >= cfg["sig_threshold"]) / len(member_vals) * 100, 1)
        else:
            pct_positive = median_pct = pct_significant = None

        scores[h] = {
            "strength": strength,
            "thrust": thrust,
            "leadership": leadership,
            "breadth": {
                "pct_positive": pct_positive,
                "median_pct": median_pct,
                "pct_significant": pct_significant,
                "n_members": len(member_vals),
            },
        }
    return scores


def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Narratives Builder")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--taxonomy", default="data/taxonomy/narratives.json", help="Path to taxonomy JSON")
    parser.add_argument("--market-features", default="data/market_features.json",
                         help="Path to Full-Market Feature Engine output (Leadership/RS/EMA/ATR/structural_rs source)")
    parser.add_argument("--config", default="config/narrative_engine.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 YOLO Dashboard — Narratives Builder")
    print(f"   Zeit: {datetime.now().isoformat()}")
    print("=" * 60)

    cfg = load_config(args.config)
    struct_cfg = cfg["narrative_structural_v1_1"]
    rsp_cfg = cfg["rsp_benchmark"]
    RSP_TICKER = rsp_cfg["ticker"]

    narratives, universe = load_taxonomy(args.taxonomy)
    print(f"\n📋 Taxonomie: {len(narratives)} Narrative, {len(universe)} eindeutige Ticker")

    # RSP dient ausschliesslich als Benchmark (RS-Historie unten + neue
    # RSP-basierte Narrative Strength/Thrust), ist aber selbst kein
    # Narrative-Mitglied und wird aus ticker_metrics wieder entfernt, damit
    # es die Perzentil-Fallback-Logik nicht verunreinigt. V6: ersetzt SPY.
    full_universe = set(universe) | {RSP_TICKER}

    # V1.1 point 17: reuse the shared V1.1 price-history cache written by
    # build_market_features.py (260 sessions, 0 extra API calls) so 3M/6M
    # structural horizons are available; only fall back to this script's own
    # short (55-session) walk-back if that cache is absent/too short.
    prices, trading_days = load_shared_price_cache_frame(cfg["market_history_cache"]["path"], full_universe)
    if prices is None:
        per_ticker, trading_days = fetch_grouped_history(full_universe)
        if len(trading_days) < 10:
            print("FATAL: Zu wenige Handelstage geladen, breche ab.", file=sys.stderr)
            sys.exit(1)
        prices = build_price_frame(per_ticker, trading_days)

    ticker_metrics, daily_ret = calc_ticker_metrics(prices)
    ticker_metrics.pop(RSP_TICKER, None)
    print(f"  ✅ Metriken für {len(ticker_metrics)}/{len(universe)} Ticker berechnet")

    market_features = load_market_features(args.market_features)
    pct_field_by_horizon = {
        "1d": "d1_pct", "1w": "w1_pct", "1m": "m1_pct", "3m": "return_3m", "6m": "return_6m",
    }
    mf_rs_field_by_horizon = {
        "1d": "rs_percentile_1d", "1w": "rs_percentile_1w", "1m": "rs_percentile_1m",
        "3m": "rs_percentile_3m", "6m": "rs_percentile_6m",
    }

    if market_features is not None:
        # Full-Market RS: percentile against the whole eligible US stock
        # universe (point 24 of V1 — the one required Leadership formula
        # change; 3M/6M added here for V1.1's structural horizons).
        percentiles_by_horizon = {
            h: {sym: market_features[sym].get(mf_field) for sym in universe if sym in market_features}
            for h, mf_field in mf_rs_field_by_horizon.items()
        }
    else:
        # Fallback: percentile against just the curated taxonomy universe
        # (pre-migration behaviour), so this script still works standalone.
        percentiles_by_horizon = {h: percentile_ranks(ticker_metrics, f) for h, f in pct_field_by_horizon.items()}

    for sym, m in ticker_metrics.items():
        m["percentile_1d"] = percentiles_by_horizon["1d"].get(sym)
        m["percentile_1w"] = percentiles_by_horizon["1w"].get(sym)
        m["percentile_1m"] = percentiles_by_horizon["1m"].get(sym)
        # V6.1 point 15: 3M/6M Stock-RS-Percentiles, previously missing from
        # the member payload (percentiles_by_horizon already computes them
        # via mf_rs_field_by_horizon -- just wasn't copied onto ticker_metrics).
        m["percentile_3m"] = percentiles_by_horizon["3m"].get(sym)
        m["percentile_6m"] = percentiles_by_horizon["6m"].get(sym)
        mf = (market_features or {}).get(sym, {})
        m["ema10_distance_pct"] = mf.get("ema10_distance_pct")
        m["ema20_distance_pct"] = mf.get("ema20_distance_pct")
        m["atr"] = mf.get("atr")
        m["atr_extension"] = mf.get("atr_extension")
        m["eligible"] = mf.get("eligible")
        m["structural_rs"] = mf.get("structural_rs")
        m["trend_strength"] = mf.get("trend_strength")
        # V6.1 point 4: market_cap for aggregate_market_cap (tie-breaker
        # only, never fed into any RS/Thrust formula). Point 16: stock_thrust_rs
        # computed upstream in build_market_features.py (same eligible
        # cross-sectional population as rs_percentile_1w/1m), passed through
        # unchanged -- no second Stock-RS/Thrust engine here.
        m["market_cap"] = mf.get("market_cap")
        m["stock_thrust_rs"] = mf.get("stock_thrust_rs")
        # RVOL/Screener/Benchmark/Futures Patch point 7: pass through the
        # already-computed volume/rvol_50 from market_features.json (no
        # second computation here) for the Member-Detailtabelle's new
        # "Volumen (M)" column.
        m["volume"] = mf.get("volume")
        m["rvol_50"] = mf.get("rvol_50")

    # Full-Universe spec point 8: every narrative-level metric (Strength/
    # Thrust/Breadth/Leadership/Structural Leadership/Trend Participation/
    # Structural Score) uses ONLY members that are BOTH semantically
    # classified into this narrative AND currently market_features.eligible
    # — a historically-classified-but-now-ineligible ticker keeps its
    # taxonomy membership (point 7) but stops moving any narrative number.
    # None (not a set) when market_features itself is unavailable: the
    # existing curated-universe fallback mode has no eligibility concept to
    # filter on, so it degrades to "use every taxonomy member", same as before.
    eligible_set = None if market_features is None else \
        {sym for sym, t in market_features.items() if t.get("eligible")}

    # Quality Patch section 8/19-21: a narrative only counts as an ACTIVE
    # dashboard basket once it has at least minimum_active_narrative_members
    # CURRENTLY ELIGIBLE members — deterministic, recomputed every run from
    # today's eligible set (never set by the LLM). Below that it's
    # "undersized": the semantic classification stays fully persisted in the
    # taxonomy (untouched here), it just doesn't get a Structural
    # Score/Lifecycle/Rank published or count toward Market Regime's
    # Narrative Environment (both consume ONLY output_narratives below) and
    # doesn't show up as a selectable narrative in Opportunities (whose
    # narrative_membership_idx is also built purely from output_narratives —
    # see build_dashboard_states.py). market_features unavailable (None)
    # degrades to "no size gate" (min_active=0), same graceful-degradation
    # rule as the eligible-filter above — there is no eligible concept to
    # gate on in that fallback mode.
    min_active_members = cfg["classification"]["minimum_active_narrative_members"] if eligible_set is not None else 0

    # Pass 1: per-narrative basket scores (unchanged existing methodology,
    # now also covering the new 3M/6M horizons) + raw structural inputs.
    narrative_rows = []
    undersized_narratives = []
    for n in narratives:
        members = [t for t in n["tickers"]
                   if t in ticker_metrics and (eligible_set is None or t in eligible_set)]
        if len(members) < min_active_members:
            undersized_narratives.append({"id": n["id"], "name": n["name"], "eligible_member_count": len(members)})
            continue
        if not members:
            print(f"  ⚠ {n['name']}: keine eligible Mitglieder mit Daten, übersprungen")
            continue
        scores = calc_basket_scores(members, ticker_metrics, daily_ret, pct_field_by_horizon, percentiles_by_horizon)
        pct_above_sma50, pct_rising_sma50, pct_above_rising_sma50 = calc_trend_participation(members, market_features)
        structural_leadership_pct = calc_structural_leadership_pct(
            members, market_features, struct_cfg["structural_leadership_rs_threshold"])
        narrative_rows.append({
            "id": n["id"], "name": n["name"], "status": n.get("status", "active"),
            "members": members, "scores": scores, "membership_meta": n.get("membership_meta", {}),
            "pct_above_sma50": pct_above_sma50, "pct_rising_sma50": pct_rising_sma50,
            "pct_above_rising_sma50": pct_above_rising_sma50,
            "structural_leadership_pct": structural_leadership_pct,
        })

    # Pass 2: narrative-vs-narrative percentiles (V1.1 point 18/19) — each
    # narrative's own Strength ranked against ALL OTHER NARRATIVES (not
    # tickers), same percentile_ranks() primitive used for ticker RS above.
    strength_percentile_by_horizon = {}
    for h in ("1m", "3m", "6m"):
        metrics = {row["id"]: {"s": row["scores"][h]["strength"]} for row in narrative_rows}
        strength_percentile_by_horizon[h] = percentile_ranks(metrics, "s")
    thrust_metrics_1w = {row["id"]: {"t": row["scores"]["1w"]["thrust"]} for row in narrative_rows}
    thrust_percentile_1w = percentile_ranks(thrust_metrics_1w, "t")

    # V6: RSP-based Narrative Strength/Thrust — single canonical RSP close
    # series (same column build_benchmark_rsp_series() reads below), shared
    # by every narrative's relative-strength line (point 29B: exactly ONE
    # canonical RSP data source).
    benchmark_close = prices[RSP_TICKER] if RSP_TICKER in prices.columns else None
    if benchmark_close is None:
        print(f"  ⚠ {RSP_TICKER} nicht in den geladenen Kursdaten enthalten — "
              "RSP-Narrative-Strength/Thrust uebersprungen (narrative_rs/thrust_rsp bleiben None)", file=sys.stderr)

    # V6.1 (Narrative Ranking & UI Bugfix Patch) point 5-7: build every
    # active narrative's relative-strength line FIRST (Pass 2a), THEN rank
    # them all cross-sectionally per horizon (Pass 2b) -- this is the actual
    # fix for the old self-window scaling bug (V6 ranked each narrative only
    # against its OWN trailing 1W/1M/3M/6M history, which with as few as 5
    # observations for 1W collapsed onto a handful of discrete values like
    # 20/40/60/80/100 and let several narratives tie at 100 simultaneously).
    # Narrative RS now answers "how strong is this narrative RIGHT NOW
    # relative to every OTHER currently active narrative", not "where does
    # today sit within this one narrative's own recent history".
    relative_strength_by_id = {}
    if benchmark_close is not None:
        for row in narrative_rows:
            eq_weight_ret = compute_narrative_equal_weight_return_series(daily_ret, row["members"])
            narrative_index = build_synthetic_narrative_index(eq_weight_ret)
            relative_strength = compute_relative_strength_line(narrative_index, benchmark_close)
            if not relative_strength.empty:
                relative_strength_by_id[row["id"]] = relative_strength

    narrative_rs, relative_performance_rsp = compute_narrative_rs(
        relative_strength_by_id, rsp_cfg["strength_windows_sessions"], sessions_ago=0)
    # Point 9: historical RS1W as of `thrust_delta_sessions` sessions ago,
    # reconstructed via the SAME cross-sectional method at that earlier
    # reference point (never the old self-window percentile) -- feeds Thrust's
    # acceleration term below.
    narrative_rs_1w_n_ago, _ = compute_narrative_rs(
        relative_strength_by_id, {"1w": rsp_cfg["strength_windows_sessions"]["1w"]},
        sessions_ago=rsp_cfg["thrust_delta_sessions"])
    thrust_rsp_by_id = {
        nid: compute_narrative_thrust_rsp(
            narrative_rs["1w"].get(nid), narrative_rs["1m"].get(nid),
            narrative_rs_1w_n_ago["1w"].get(nid), rsp_cfg["thrust_weights"])
        for nid in relative_strength_by_id
    }
    if relative_strength_by_id:
        n_distinct_1w = len({v for v in narrative_rs["1w"].values() if v is not None})
        print(f"  ✅ Narrative RS (cross-sectional): {len(relative_strength_by_id)} aktive Narrative gerankt, "
              f"{n_distinct_1w} unterschiedliche RS-1W-Werte (kein 20/40/60/80/100-Zwang mehr)")

    output_narratives = []
    for row in narrative_rows:
        nid, members, scores = row["id"], row["members"], row["scores"]

        narrative_rs_for_id = {label: narrative_rs[label].get(nid) for label in rsp_cfg["strength_windows_sessions"]}
        relative_performance_for_id = {label: relative_performance_rsp[label].get(nid)
                                        for label in rsp_cfg["strength_windows_sessions"]}
        thrust_rsp = thrust_rsp_by_id.get(nid)

        # V6.1 point 4: aggregate_market_cap — sum of CURRENTLY ELIGIBLE
        # members' market_cap (each ticker counted once per narrative;
        # overlapping membership across DIFFERENT narratives stays allowed,
        # point 4's explicit rule). Tie-breaker ONLY -- never an input to
        # narrative_rs/thrust_rsp above. None (not 0) if no member has a
        # known market_cap, same graceful-degradation convention as
        # everywhere else in this pipeline.
        member_caps = [ticker_metrics[m]["market_cap"] for m in members
                       if ticker_metrics[m].get("market_cap") is not None]
        aggregate_market_cap = round(sum(member_caps), 2) if member_caps else None

        strength_percentile_1m = strength_percentile_by_horizon["1m"].get(nid)
        strength_percentile_3m = strength_percentile_by_horizon["3m"].get(nid)
        strength_percentile_6m = strength_percentile_by_horizon["6m"].get(nid)

        structural_price_strength = clamp_0_100(renormalized_weighted_sum(
            {
                "strength_percentile_1m": strength_percentile_1m,
                "strength_percentile_3m": strength_percentile_3m,
                "strength_percentile_6m": strength_percentile_6m,
            },
            struct_cfg["structural_price_strength_weights"]))

        narrative_structural_score = clamp_0_100(renormalized_weighted_sum(
            {
                "structural_price_strength": structural_price_strength,
                "trend_participation": row["pct_above_rising_sma50"],
                "structural_leadership_pct": row["structural_leadership_pct"],
                "breadth_pct_positive_1m": scores["1m"]["breadth"]["pct_positive"],
            },
            struct_cfg["score_weights"]))

        momentum_modifier = calc_momentum_modifier(
            scores["1w"]["thrust"], thrust_percentile_1w.get(nid),
            narrative_structural_score, struct_cfg["momentum_modifier"])

        output_narratives.append({
            "id": nid,
            "name": row["name"],
            "status": row["status"],
            "n_members": len(members),
            "scores": scores,
            # Shallow copy per narrative (not the shared ticker_metrics dict
            # itself) — a ticker's assignment_priority is per-(narrative,
            # ticker), e.g. "primary" here but "secondary" in another
            # narrative this same ticker also belongs to (point 10/11).
            "members": [{**ticker_metrics[m],
                         "assignment_priority": row["membership_meta"].get(m, {}).get("assignment_priority")}
                        for m in members],
            # V1.1 point 17-22: Structural Score replaces the old 1W-heavy
            # Momentum Score as the primary narrative ranking metric. Thrust
            # stays visible inside `scores` (per-horizon) but is no longer a
            # component of this score — see momentum_modifier for its role.
            "structural_price_strength": structural_price_strength,
            "strength_percentile_1m": strength_percentile_1m,
            "strength_percentile_3m": strength_percentile_3m,
            "strength_percentile_6m": strength_percentile_6m,
            "trend_participation": {
                "pct_above_sma50": row["pct_above_sma50"],
                "pct_rising_sma50": row["pct_rising_sma50"],
                "pct_above_rising_sma50": row["pct_above_rising_sma50"],
            },
            "structural_leadership_pct": row["structural_leadership_pct"],
            "narrative_structural_score": narrative_structural_score,
            "momentum_modifier": momentum_modifier,
            # Narrative-vs-narrative 1W Thrust percentile — used by
            # build_dashboard_states.py's structural Lifecycle EMERGING
            # condition (thrust_percentile_1w_min), NOT a narrative_structural_score
            # component (Thrust stays out of the structural score itself).
            "thrust_percentile_1w": thrust_percentile_1w.get(nid),
            # V6.1 point 6-8: the headline metrics, now CROSS-SECTIONAL
            # (point 5-7 — see compute_narrative_rs/cross_sectional_percentile_ranks
            # above for why this replaces V6's self-window strength_rsp).
            # "Jeff-inspired Relative Strength gegen RSP" / "Jeff-inspired
            # Thrust candidate v1" remain candidate formulas, not a
            # reproduction of Jeff Sun's unpublished original. The four
            # narrative_rs timeframes stay fully separate (no averaging/
            # composite, point 7). relative_performance_rsp carries the RAW
            # ratio-based return behind each narrative_rs percentile (point 8).
            "narrative_rs": narrative_rs_for_id,
            "relative_performance_rsp": relative_performance_for_id,
            "thrust_rsp": thrust_rsp,
            "aggregate_market_cap": aggregate_market_cap,
            # Point 8's explicit backward-compat allowance: "strength_rsp"
            # kept as a DEPRECATED ALIAS of narrative_rs (the SAME dict, not
            # a second calculation) for any consumer still reading the old
            # V6 key name. New consumers (frontend included, point 8) read
            # narrative_rs exclusively.
            "strength_rsp": narrative_rs_for_id,
        })
        print(f"  ✅ {row['name']}: {len(members)} Ticker | Structural Score {narrative_structural_score} | "
              f"Trend Participation {row['pct_above_rising_sma50']} | Modifier {momentum_modifier} | "
              f"RS 1W={narrative_rs_for_id['1w']} 1M={narrative_rs_for_id['1m']} | Thrust={thrust_rsp} | "
              f"Aggregate Market Cap={aggregate_market_cap}")

    # V6.1 point 19: Healthcare/Biotech Sanity Gate -- runs on the FINAL
    # active/eligible-filtered output_narratives, so it only ever sees what
    # the dashboard would actually show. A violation fails the build loudly
    # (sys.exit(1) with full diagnostics) rather than silently shipping a
    # Healthcare/Biotech narrative that slipped past the upstream
    # classify_healthcare_biotech exclusion in build_market_features.py.
    hc_gate_cfg = cfg.get("universe", {}).get("healthcare_biotech_sanity_gate", {})
    audit_keywords = hc_gate_cfg.get("audit_keywords", [])
    hc_leaks = find_healthcare_biotech_leaks(output_narratives, market_features, audit_keywords, min_active_members)
    if hc_leaks:
        print("\n❌ Healthcare/Biotech Sanity Gate FEHLGESCHLAGEN (Punkt 19) — Build abgebrochen:", file=sys.stderr)
        for v in hc_leaks:
            print(f"  Narrative '{v['narrative_name']}' ({v['narrative_id']}): "
                  f"{v['eligible_member_count']} eligible Mitglieder, "
                  f"Name-Treffer: {v['name_matched_keywords']}, "
                  f"{v['flagged_member_count']} Mitglieder mit Treffer", file=sys.stderr)
            for fm in v["flagged_members"]:
                print(f"    {fm['symbol']}: SIC {fm['sic_code']} ({fm['sic_description']}) — "
                      f"Treffer {fm['matched_keywords']} — company_description: {fm['company_description']!r}",
                      file=sys.stderr)
        sys.exit(1)
    print(f"  ✅ Healthcare/Biotech Sanity Gate: 0 aktive Narrative mit >= {min_active_members} "
          f"eligible Mitgliedern lesen sich als Healthcare/Biotech")

    rs_history = compute_narrative_rs_history(
        narrative_rows, relative_strength_by_id, trading_days, benchmark_ticker=RSP_TICKER)
    if rs_history is not None:
        print(f"  ✅ Benchmark-RS-Historie: {len(rs_history['narratives'])} Narrative x "
              f"{rs_history['lookback_trading_days']} Handelstage vs. {RSP_TICKER}")

    benchmark_rsp = build_benchmark_rsp_series(prices, RSP_TICKER)
    if benchmark_rsp is not None:
        print(f"  ✅ Benchmark · RSP Serie: {len(benchmark_rsp['dates'])} Handelstage "
              f"({benchmark_rsp['dates'][0]} .. {benchmark_rsp['dates'][-1]})")
    else:
        print(f"  ⚠ {RSP_TICKER}-Preis-Serie nicht verfuegbar — Benchmark-Chart-Daten fehlen diesen Lauf", file=sys.stderr)

    # Full-Universe spec point 23: distinguish the WHOLE market's eligible
    # universe (eligible_universe_size, from market_features — independent
    # of narrative classification) from what's actually classified+eligible
    # right now. total_active_memberships can exceed
    # unique_classified_eligible_members because Secondary memberships are
    # real, counted memberships, not a subset (point 23: "das ist korrekt").
    unique_classified_eligible = {m["symbol"] for row in output_narratives for m in row["members"]}
    total_active_memberships = sum(row["n_members"] for row in output_narratives)
    eligible_universe_size = len(eligible_set) if eligible_set is not None else len(universe)
    coverage_pct = round(len(unique_classified_eligible) / eligible_universe_size * 100, 1) \
        if eligible_universe_size else 100.0

    # Quality Patch section 22-23: successor coverage metrics — how many
    # eligible tickers actually have an ACTIVE (>= min_active_members
    # eligible members) Primary Narrative, vs. classified-into-SOMETHING
    # (unique_classified_eligible_members above, which also counts
    # undersized/secondary-only membership). A ticker without an active
    # Primary is an explicitly VALID outcome now (never forced) — see
    # scripts/reconcile_narrative_universe.py's confidence-gated
    # classification and Opportunities' narrative-independent coverage.
    active_ids = {row["id"] for row in output_narratives}
    primary_narrative_of = {}
    for n in narratives:
        for sym, meta in n.get("membership_meta", {}).items():
            if meta.get("assignment_priority") == "primary":
                primary_narrative_of[sym] = n["id"]
    eligible_pop = eligible_set if eligible_set is not None else set(universe)
    eligible_with_active_narrative = sum(1 for t in eligible_pop if primary_narrative_of.get(t) in active_ids)
    eligible_without_active_narrative = len(eligible_pop) - eligible_with_active_narrative
    active_narrative_coverage_pct = round(eligible_with_active_narrative / len(eligible_pop) * 100, 1) \
        if eligible_pop else 100.0

    output = {
        "meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Massive (api.massive.com)",
            "trading_days_used": len(trading_days),
            "date_range": [trading_days[0], trading_days[-1]] if trading_days else None,
            "universe_size": len(universe),
            "eligible_universe_size": eligible_universe_size,
            "unique_classified_eligible_members": len(unique_classified_eligible),
            "total_active_memberships": total_active_memberships,
            "coverage_pct": coverage_pct,
            "eligible_with_active_narrative": eligible_with_active_narrative,
            "eligible_without_active_narrative": eligible_without_active_narrative,
            "active_narrative_coverage_pct": active_narrative_coverage_pct,
            "active_narrative_count": len(active_ids),
            "undersized_narrative_count": len(undersized_narratives),
            "undersized_narratives": sorted(undersized_narratives, key=lambda u: -u["eligible_member_count"]),
        },
        "narratives": output_narratives,
        "rs_history": rs_history,
        "benchmark_rsp": benchmark_rsp,
    }

    with open(out_dir / "narratives.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Narratives geschrieben → {out_dir / 'narratives.json'}")
    print(f"   Aktive Narrative: {len(output_narratives)} | Undersized (< {min_active_members} eligible): "
          f"{len(undersized_narratives)} | Ticker gesamt: {len(ticker_metrics)}")
    print(f"   Active Narrative Coverage: {eligible_with_active_narrative}/{len(eligible_pop)} "
          f"({active_narrative_coverage_pct}%) eligible Ticker mit aktivem Primary Narrative")
    print("=" * 60)


if __name__ == "__main__":
    main()
