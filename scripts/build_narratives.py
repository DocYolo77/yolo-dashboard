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
SPY is folded into the same grouped-daily walk (zero extra API calls — the
grouped-daily response already contains the whole market per day) purely as
the comparison series, then popped back out of ticker_metrics so it never
appears as a narrative member or pollutes the percentile-fallback pool. See
compute_narrative_rs_history() for the basket-vs-SPY cumulative-return
calculation and data/narratives.json's "rs_history" key for the output
shape (one shared "dates" array + per-narrative value lists — checked
against real narrative counts, this is a few KB, not worth a separate file).
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

        out[sym] = {
            "symbol": sym,
            "price": round(float(last), 2),
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


def compute_narrative_rs_history(narratives, daily_ret, trading_days, eligible_set=None, lookback_days=RS_HISTORY_LOOKBACK_DAYS):
    """Basket-vs-SPY relative-strength time series per narrative, for the
    dashboard's 'Benchmark' comparison chart. Same basket daily-return series
    as Strength/Thrust (basket_daily_return_series), compounded from the
    start of the lookback window; relative strength = compounded basket
    return minus compounded SPY return, in percentage points, so SPY is
    always the flat 0% baseline the chart draws as a dashed line. Returns
    None (with a warning) if SPY wasn't in the fetched universe or has too
    little history — the frontend degrades to an empty-state message rather
    than crashing, matching load_market_features' graceful-fallback pattern.

    `eligible_set` restricts basket membership the SAME way as the main
    Strength/Thrust/Breadth/Leadership calculation below (Full-Universe spec
    point 8) — a historically-classified-but-currently-ineligible ticker
    must not silently keep steering the Benchmark chart either. None (the
    default) means "no eligibility filter" — same graceful-degradation
    fallback as everywhere market_features may be unavailable."""
    if "SPY" not in daily_ret.columns:
        print("  ⚠ SPY nicht in den geladenen Kursdaten enthalten — Benchmark-RS-Historie übersprungen", file=sys.stderr)
        return None

    window_dates = trading_days[-lookback_days:]
    spy_ret = daily_ret["SPY"].reindex(window_dates)
    if spy_ret.dropna().shape[0] < 10:
        print("  ⚠ Zu wenig SPY-Historie für Benchmark-RS-Historie — übersprungen", file=sys.stderr)
        return None
    spy_cum = (1 + spy_ret.fillna(0) / 100).cumprod() - 1

    series_by_narrative = {}
    for n in narratives:
        members = [t for t in n["tickers"]
                   if t in daily_ret.columns and (eligible_set is None or t in eligible_set)]
        if not members:
            continue
        basket_ret = basket_daily_return_series(daily_ret, members).reindex(window_dates)
        if basket_ret.dropna().shape[0] < 10:
            continue
        basket_cum = (1 + basket_ret.fillna(0) / 100).cumprod() - 1
        relative = (basket_cum - spy_cum) * 100
        series_by_narrative[n["id"]] = [round(float(v), 2) for v in relative.tolist()]

    dates_fmt = [datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.") for d in window_dates]
    return {
        "benchmark": "SPY",
        "lookback_trading_days": len(window_dates),
        "dates": dates_fmt,
        "narratives": series_by_narrative,
    }


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

    narratives, universe = load_taxonomy(args.taxonomy)
    print(f"\n📋 Taxonomie: {len(narratives)} Narrative, {len(universe)} eindeutige Ticker")

    # SPY dient ausschliesslich als Benchmark fuer die RS-Historie unten, ist
    # aber selbst kein Narrative-Mitglied und wird aus ticker_metrics wieder
    # entfernt, damit es die Perzentil-Fallback-Logik nicht verunreinigt.
    full_universe = set(universe) | {"SPY"}

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
    ticker_metrics.pop("SPY", None)
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
        mf = (market_features or {}).get(sym, {})
        m["ema10_distance_pct"] = mf.get("ema10_distance_pct")
        m["ema20_distance_pct"] = mf.get("ema20_distance_pct")
        m["atr"] = mf.get("atr")
        m["atr_extension"] = mf.get("atr_extension")
        m["eligible"] = mf.get("eligible")
        m["structural_rs"] = mf.get("structural_rs")
        m["trend_strength"] = mf.get("trend_strength")

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

    # Pass 1: per-narrative basket scores (unchanged existing methodology,
    # now also covering the new 3M/6M horizons) + raw structural inputs.
    narrative_rows = []
    for n in narratives:
        members = [t for t in n["tickers"]
                   if t in ticker_metrics and (eligible_set is None or t in eligible_set)]
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

    output_narratives = []
    for row in narrative_rows:
        nid, members, scores = row["id"], row["members"], row["scores"]

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
        })
        print(f"  ✅ {row['name']}: {len(members)} Ticker | Structural Score {narrative_structural_score} | "
              f"Trend Participation {row['pct_above_rising_sma50']} | Modifier {momentum_modifier}")

    rs_history = compute_narrative_rs_history(
        narratives, daily_ret, trading_days,
        eligible_set if eligible_set is not None else set(universe))
    if rs_history is not None:
        print(f"  ✅ Benchmark-RS-Historie: {len(rs_history['narratives'])} Narrative x "
              f"{rs_history['lookback_trading_days']} Handelstage vs. SPY")

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
        },
        "narratives": output_narratives,
        "rs_history": rs_history,
    }

    with open(out_dir / "narratives.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Narratives geschrieben → {out_dir / 'narratives.json'}")
    print(f"   Narrative: {len(output_narratives)} | Ticker gesamt: {len(ticker_metrics)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
