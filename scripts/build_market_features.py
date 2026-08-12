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
  50-day simple moving average. SMA50 needs 50 trading days of history,
  hence TRADING_DAYS_NEEDED below is 60, not 35.
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
from build_narratives import percentile_ranks, HORIZONS  # noqa: E402  (reuse canonical logic)
import build_market_reference as ref  # noqa: E402

MASSIVE_BASE = "https://api.massive.com"
TRADING_DAYS_NEEDED = 60  # SMA50 (ATR-Extension reference MA) needs 50 closes + buffer
MAX_CALENDAR_LOOKBACK = 100
MIN_RESULTS_FOR_TRADING_DAY = 1000
ADR_LOOKBACK_DEFAULT = 20


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

def fetch_grouped_history_full_market(universe_set):
    """Walk backward day by day, collecting grouped-daily OHLC for tickers in
    `universe_set` until we have TRADING_DAYS_NEEDED trading days. One API
    call covers the whole market per day (same pattern as
    build_narratives.fetch_grouped_history), so cost does not scale with the
    size of `universe_set`."""
    print(f"\n📊 Lade marktweite Grouped-Daily-OHLC (Ziel: {TRADING_DAYS_NEEDED} Handelstage)...")
    per_ticker_close = {t: {} for t in universe_set}
    per_ticker_high = {t: {} for t in universe_set}
    per_ticker_low = {t: {} for t in universe_set}
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
            if sym in per_ticker_close:
                per_ticker_close[sym][date_str] = row.get("c")
                per_ticker_high[sym][date_str] = row.get("h")
                per_ticker_low[sym][date_str] = row.get("l")
        print(f"  → {date_str}: {data['resultsCount']} Ticker (Handelstag {len(trading_days)}/{TRADING_DAYS_NEEDED})")

    trading_days.sort()
    print(f"  ✅ {len(trading_days)} Handelstage geladen ({calendar_checked} Kalendertage geprueft)")
    return per_ticker_close, per_ticker_high, per_ticker_low, trading_days


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


def calc_ticker_features(close_df, high_df, low_df, adr_lookback):
    """Returns {symbol: {...}} with price/ADR20/EMA/ATR/performance fields."""
    out = {}
    for sym in close_df.columns:
        close = close_df[sym].dropna()
        if len(close) < max(adr_lookback, 51):  # 50 for SMA50 + 1 for the pct_ago(50) edge
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
        ema10_distance_pct = round(float((last - ema10) / ema10 * 100), 2)
        ema20_distance_pct = round(float((last - ema20) / ema20 * 100), 2)
        sma50 = close.rolling(50).mean().iloc[-1]

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

        out[sym] = {
            "symbol": sym,
            "close": round(float(last), 2),
            "adr20": adr20,
            "sma50": round(float(sma50), 2) if sma50 and not np.isnan(sma50) else None,
            "gain_from_sma50_pct": gain_from_sma50_pct,
            "d1_pct": pct_ago(1),
            "w1_pct": pct_ago(5),
            "m1_pct": pct_ago(21),
            "ema10": round(float(ema10), 2),
            "ema20": round(float(ema20), 2),
            "ema10_distance_pct": ema10_distance_pct,
            "ema20_distance_pct": ema20_distance_pct,
            "atr": atr,
            "atr_extension": atr_extension,
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


def compute_eligibility(adr20, market_cap, universe_cfg):
    """Universe-Filter (point 2): ADR20 > adr_minimum_pct AND market_cap >=
    market_cap_minimum_usd. Both must be known (not None) — missing data is
    NOT eligible-by-default. Extracted as a pure function so it is testable
    without a market-wide fetch."""
    adr_ok = adr20 is not None and adr20 > universe_cfg["adr_minimum_pct"]
    cap_ok = market_cap is not None and market_cap >= universe_cfg["market_cap_minimum_usd"]
    return bool(adr_ok and cap_ok)


def compute_eligible_universe(features, market_cap_by_symbol, universe_cfg):
    """Resolve eligibility for every ticker with a computed feature set.
    MUST run before any percentile ranking (V1 rebuild point 7 fix):
    RS/Thrust percentiles are only meaningful when ranked against tickers
    that actually pass the Universe Filter, not against every ticker that
    merely has enough price history to compute a feature."""
    return {sym: compute_eligibility(f["adr20"], market_cap_by_symbol.get(sym), universe_cfg)
            for sym, f in features.items()}


def eligible_percentile_ranks(features, eligible_by_symbol, field):
    """percentile_ranks() restricted to the eligible subset (point 7 fix).
    Non-eligible tickers are excluded from the ranking pool entirely — they
    get no percentile (None downstream), not a 0/bottom-of-market rank."""
    eligible_features = {sym: f for sym, f in features.items() if eligible_by_symbol.get(sym)}
    return percentile_ranks(eligible_features, field)


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

    types_ref = load_types_reference(taxonomy_dir / "market_reference_types.json")
    type_universe = type_eligible_universe(types_ref, set(u_cfg["excluded_types"]))
    print(f"\n📋 {len(type_universe)} Ticker nach Asset-Type-Filter (ETF/ETN/FUND/... ausgeschlossen)")

    close_hist, high_hist, low_hist, trading_days = fetch_grouped_history_full_market(type_universe)
    if len(trading_days) < 21:
        print("FATAL: Zu wenige Handelstage geladen, breche ab.", file=sys.stderr)
        sys.exit(1)

    close_df = build_frame(close_hist, trading_days, type_universe)
    high_df = build_frame(high_hist, trading_days, type_universe)
    low_df = build_frame(low_hist, trading_days, type_universe)
    daily_ret = close_df.pct_change() * 100

    features = calc_ticker_features(close_df, high_df, low_df, u_cfg["adr_lookback_sessions"])
    print(f"  ✅ Preis-/ADR-/EMA-/ATR-Features fuer {len(features)}/{len(type_universe)} Ticker berechnet")

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

    # Eligibility MUST be resolved BEFORE any percentile ranking (point 7
    # fix) — see compute_eligible_universe()/eligible_percentile_ranks().
    eligible_by_symbol = compute_eligible_universe(features, market_cap_by_symbol, u_cfg)
    n_eligible = sum(1 for v in eligible_by_symbol.values() if v)
    print(f"  ✅ {n_eligible}/{len(features)} Ticker eligible (ADR{u_cfg['adr_lookback_sessions']} > "
          f"{u_cfg['adr_minimum_pct']}% UND Market Cap >= ${u_cfg['market_cap_minimum_usd']:,.0f}) — "
          f"RS-/Thrust-Perzentile werden NUR gegen dieses Subset berechnet")

    # RS percentiles across the eligible universe ONLY — this is the
    # "Full-Market RS" the Leadership score (and everything downstream:
    # Discovery Candidates, Market Regime Momentum, Opportunities) relies
    # on. Non-eligible tickers get rs_percentile_* = None: not ranked, never
    # silently treated as 0 (would misrepresent them as market-bottom).
    pct_field_by_horizon = {"1d": "d1_pct", "1w": "w1_pct", "1m": "m1_pct"}
    percentiles_by_horizon = {h: eligible_percentile_ranks(features, eligible_by_symbol, f)
                               for h, f in pct_field_by_horizon.items()}

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

        rs_1w = percentiles_by_horizon["1w"].get(sym)
        rs_1m = percentiles_by_horizon["1m"].get(sym)
        thrust_1d_pct = thrust_percentiles["1d"].get(sym)
        thrust_1w_pct = thrust_percentiles["1w"].get(sym)
        thrust_1m_pct = thrust_percentiles["1m"].get(sym)

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
            "thrust_1d": thrust_by_horizon["1d"].get(sym),
            "thrust_1w": thrust_by_horizon["1w"].get(sym),
            "thrust_1m": thrust_by_horizon["1m"].get(sym),
            "thrust_percentile_1d": thrust_1d_pct,
            "thrust_percentile_1w": thrust_1w_pct,
            "thrust_percentile_1m": thrust_1m_pct,
            "sic_code": overview.get("sic_code"),
            "sic_description": overview.get("sic_description"),
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
            "eligible_count": n_eligible,
            "discovery_candidate_count": sum(1 for t in output_tickers.values() if t["discovery_candidate"]),
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
