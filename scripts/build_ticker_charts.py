#!/usr/bin/env python3
"""
YOLO Dashboard — Ticker Hover-Chart Data Builder
Writes data/ticker_charts.json: a compact, per-ticker OHLC + EMA10/EMA20/
SMA50/SMA200 window for every currently ELIGIBLE ticker, so the frontend's
hover mini-chart (Opportunities table + narrative member tables) never has
to hit an API on hover — the same rolling price-history cache
build_market_features.py already maintains (260 trading days, close/high/
low/open) is reused here with ZERO additional API calls.

Window size (WINDOW_DAYS = 60): SMA200 needs 200 trading days of history to
be valid at any given point. With a 260-session rolling cache, the oldest
day for which SMA200 can be fully computed without truncation is
260 - 200 = 60 sessions back — i.e. 60 is the largest display window for
which SMA200 is guaranteed populated across the ENTIRE visible range, not
just its most recent portion. This lands squarely in the user-requested
"60-80 Handelstage" range, at the safe end of it.

Indicator conventions match the rest of the dashboard exactly (no new
formula invented here):
  EMA10/EMA20 -> close.ewm(span=n).mean(), same as
                 build_market_features.calc_ticker_features (the values
                 already shown as ema10_distance_pct/ema20_distance_pct
                 elsewhere on the dashboard come from this same call).
  SMA50       -> close.rolling(50).mean(), same as build_market_features.py.
  SMA200      -> close.rolling(200).mean() — new (no scalar SMA200 field
                 exists elsewhere on the dashboard yet), but the identical
                 simple-rolling-mean convention as SMA50.

A ticker's close series is dropna()'d first (same approach as
calc_ticker_features) before computing indicators, so leading "not yet
trading"/gap days never pollute the EMA/SMA warm-up; the emitted `dates`
array is that ticker's OWN trading-day index for its trailing window, not a
market-wide date it may not have traded on.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WINDOW_DAYS = 60
MIN_BARS = 5  # a ticker with less history than this has no meaningful chart


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_eligible_set(market_features):
    return {sym for sym, t in market_features.get("tickers", {}).items() if t.get("eligible")}


def _series_out(s, window_idx):
    aligned = s.reindex(window_idx)
    return [round(float(v), 2) if pd.notna(v) else None for v in aligned]


def compute_ticker_chart(dates, close_vals, high_vals, low_vals, open_vals, window_days=WINDOW_DAYS):
    """Returns {"dates":[...], "o":[...], "h":[...], "l":[...], "c":[...],
    "ema10":[...], "ema20":[...], "sma50":[...], "sma200":[...]} for the
    trailing `window_days` of this ticker's OWN (dropna'd) trading history,
    or None if there isn't even MIN_BARS of usable close data."""
    idx = pd.Index(dates)
    close = pd.Series(close_vals, index=idx, dtype=float).dropna()
    if len(close) < MIN_BARS:
        return None
    high = pd.Series(high_vals, index=idx, dtype=float)
    low = pd.Series(low_vals, index=idx, dtype=float)
    openp = pd.Series(open_vals, index=idx, dtype=float)

    ema10 = close.ewm(span=10).mean()
    ema20 = close.ewm(span=20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    window_idx = close.index[-window_days:]
    return {
        "dates": list(window_idx),
        "o": _series_out(openp, window_idx),
        "h": _series_out(high, window_idx),
        "l": _series_out(low, window_idx),
        "c": _series_out(close, window_idx),
        "ema10": _series_out(ema10, window_idx),
        "ema20": _series_out(ema20, window_idx),
        "sma50": _series_out(sma50, window_idx),
        "sma200": _series_out(sma200, window_idx),
    }


def build_all_charts(eligible_tickers, cache, window_days=WINDOW_DAYS):
    dates = cache.get("dates", [])
    cached_tickers = cache.get("tickers", {})
    out = {}
    for sym in sorted(eligible_tickers):
        series = cached_tickers.get(sym)
        if not series:
            continue
        chart = compute_ticker_chart(
            dates, series.get("close", []), series.get("high", []),
            series.get("low", []), series.get("open", []), window_days)
        if chart is not None:
            out[sym] = chart
    return out


def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Ticker Hover-Chart Data Builder")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--market-features", default="data/market_features.json")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--price-cache", default=None, help="Override config.market_history_cache.path")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 YOLO Dashboard — Ticker Hover-Chart Data Builder")
    print(f"   Zeit: {datetime.now().isoformat()}")
    print("=" * 60)

    market_features = load_json(args.market_features)
    if market_features is None:
        print("FATAL: data/market_features.json fehlt.", file=sys.stderr)
        sys.exit(1)

    cfg = load_json(args.config) or {}
    cache_path = args.price_cache or (cfg.get("market_history_cache") or {}).get("path", ".cache/market_history.json")
    cache = load_json(cache_path)
    if cache is None:
        print(f"  ⚠ Preis-Cache {cache_path} nicht gefunden — schreibe leere ticker_charts.json", file=sys.stderr)
        cache = {"dates": [], "tickers": {}}

    eligible = compute_eligible_set(market_features)
    print(f"  → Eligible Universe: {len(eligible)} Ticker | Preis-Cache: {len(cache.get('dates', []))} "
          f"Handelstage, {len(cache.get('tickers', {}))} Ticker")

    charts = build_all_charts(eligible, cache, args.window_days)
    print(f"  ✅ Hover-Chart-Daten fuer {len(charts)}/{len(eligible)} eligible Ticker berechnet "
          f"({len(eligible) - len(charts)} ohne ausreichende Preis-Historie im Cache)")

    output = {
        "meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": args.window_days,
            "eligible_count": len(eligible),
            "tickers_with_charts": len(charts),
        },
        "tickers": charts,
    }

    out_path = Path(args.out_dir) / "ticker_charts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))  # compact -- bulk per-ticker arrays, not meant for human diff

    size_kb = out_path.stat().st_size / 1024
    print(f"\n✅ ticker_charts.json geschrieben → {out_path} ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
