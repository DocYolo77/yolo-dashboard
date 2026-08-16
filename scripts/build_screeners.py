#!/usr/bin/env python3
"""
YOLO Dashboard — Predefined Screeners Builder (V6, spec point 19-27)
Evaluates a small set of FIXED, config-defined Screener presets
(config/narrative_engine.json's screener_presets — explicitly NOT a free/
universal programmable screener, spec point 19) against the already-final-
filtered eligible YOLO stock universe from data/market_features.json, and
writes data/screeners.json.

No additional Massive API calls: every field a preset rule can reference
already lives in market_features.json (Full-Market Feature Engine) or
data/taxonomy/market_reference_types.json (primary_exchange for the
TradingView export). Massive stays the sole source of truth for the
underlying data; this script only re-filters/re-sorts what's already been
computed. The Massive API key is never read, referenced, or needed here —
this script has no network access at all, which is also the concrete
enforcement of "API Key niemals ins Frontend" for the Screener feature
(there is nothing key-shaped anywhere near data/screeners.json's build or
consumption path).

Base universe filters (Stocks, no Healthcare/Biotech, Market Cap >= $1B,
ADR20 > 4%) are NOT recomputed here — a ticker's `eligible` flag from
market_features.json (already computed upstream of this script, see
build_market_features.py's V6 Healthcare/Biotech-exclusion-before-
eligible=true pipeline order) is the single gate; preset rules only ever
add FURTHER conditions on top of that gate (spec point 21: "diese
Basisfilter werden NICHT hier erneut dupliziert").

Rule engine (spec point 22): each preset is `{"all": [...], "any": [...]}`
— ALL of `all` must pass AND (if `any` is non-empty) AT LEAST ONE of `any`
must pass. Six ops only: >, >=, <, <=, between, ==. Deliberately not a
general-purpose rule engine — no boolean nesting, no OR-of-AND groups, no
custom functions.

TradingView export (spec point 26): "Ticker kopieren" and "TradingView TXT"
are pure client-side/manual actions (see index.html) — this script only
prepares the `primary_exchange` (MIC) -> `tradingview_symbol` mapping data
using the explicitly audited config.tradingview_export.mic_to_exchange
table (verified against the real eligible-universe MIC distribution, see
that config block's comment). An unmapped MIC gets tradingview_symbol=null
and is listed in meta.unknown_mic_tickers for audit — NEVER silently
mis-prefixed. No automatic TradingView watchlist manipulation via scraping
or credentials anywhere in this pipeline (spec's explicit security
constraint) — the actual TXT file is only ever generated in the browser
from this already-computed data.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_rule(rule, ticker):
    """Single screener rule against one ticker's market_features.json
    record. A missing/None field value NEVER passes any rule (no
    fabricated default) — matches the graceful-degradation convention used
    throughout the rest of this pipeline (percentiles/eligibility/etc.)."""
    val = ticker.get(rule["field"])
    if val is None:
        return False
    op = rule["op"]
    if op == ">":
        return val > rule["value"]
    if op == ">=":
        return val >= rule["value"]
    if op == "<":
        return val < rule["value"]
    if op == "<=":
        return val <= rule["value"]
    if op == "==":
        return val == rule["value"]
    if op == "between":
        return rule["min"] <= val <= rule["max"]
    raise ValueError(f"Unbekannter Screener-Operator: {op!r}")


def eval_preset(preset_cfg, ticker):
    """ALL of `all` (if any) must pass, AND at least one of `any` (if
    non-empty) must pass. An empty/absent `any` list imposes no extra
    condition (pure `all`-gated preset)."""
    all_rules = preset_cfg.get("all", [])
    if not all(eval_rule(r, ticker) for r in all_rules):
        return False
    any_rules = preset_cfg.get("any", [])
    if any_rules and not any(eval_rule(r, ticker) for r in any_rules):
        return False
    return True


def mic_to_tradingview_symbol(symbol, primary_exchange, mic_to_exchange):
    """(tradingview_symbol, unknown_mic) — tradingview_symbol is None
    whenever `primary_exchange` is missing or not in the explicitly
    audited mapping (spec point 26: never silently mis-prefixed)."""
    if not primary_exchange:
        return None, True
    exchange = mic_to_exchange.get(primary_exchange)
    if exchange is None:
        return None, True
    return f"{exchange}:{symbol}", False


SCREENER_TICKER_FIELDS = (
    "close", "adr20", "market_cap", "rs_percentile_1w", "rs_percentile_1m",
    "rs_percentile_3m", "rs_percentile_6m", "structural_rs",
    "sma50_distance_pct", "sma200_distance_pct", "ema10_distance_pct", "ema20_distance_pct",
)


def build_screener_ticker_row(symbol, ticker, primary_exchange, mic_to_exchange):
    tradingview_symbol, unknown_mic = mic_to_tradingview_symbol(symbol, primary_exchange, mic_to_exchange)
    row = {"symbol": symbol}
    for field in SCREENER_TICKER_FIELDS:
        row[field] = ticker.get(field)
    row["primary_exchange"] = primary_exchange
    row["tradingview_symbol"] = tradingview_symbol
    return row, unknown_mic


def weekly_strength_sort_key(row):
    """Spec point 24: (1) rs_percentile_1w desc, (2) structural_rs desc,
    (3) ticker alphabetical. No new score formula — pure multi-key sort on
    already-existing fields. None sorts last within each key (treated as
    the lowest possible value), never crashes on missing data."""
    rs = row["rs_percentile_1w"]
    structural = row["structural_rs"]
    return (-(rs if rs is not None else -1), -(structural if structural is not None else -1), row["symbol"])


def monthly_strength_sort_key(row):
    """RVOL/Screener/Benchmark/Futures Patch point 13: same multi-key shape
    as weekly_strength_sort_key, just on the 1M Strength percentile instead
    of 1W — (1) rs_percentile_1m desc, (2) structural_rs desc, (3) ticker
    alphabetical."""
    rs = row["rs_percentile_1m"]
    structural = row["structural_rs"]
    return (-(rs if rs is not None else -1), -(structural if structural is not None else -1), row["symbol"])


def three_month_strength_sort_key(row):
    """Strength Screeners 3M/6M Union Patch point 5: same multi-key shape,
    on the 3M Strength percentile — (1) rs_percentile_3m desc, (2)
    structural_rs desc, (3) ticker alphabetical."""
    rs = row["rs_percentile_3m"]
    structural = row["structural_rs"]
    return (-(rs if rs is not None else -1), -(structural if structural is not None else -1), row["symbol"])


def six_month_strength_sort_key(row):
    """Point 6: same shape, on the 6M Strength percentile."""
    rs = row["rs_percentile_6m"]
    structural = row["structural_rs"]
    return (-(rs if rs is not None else -1), -(structural if structural is not None else -1), row["symbol"])


SCREENER_SORT_KEYS = {
    "weekly_strength": weekly_strength_sort_key,
    "monthly_strength": monthly_strength_sort_key,
    "three_month_strength": three_month_strength_sort_key,
    "six_month_strength": six_month_strength_sort_key,
}


def build_screener(preset_id, preset_cfg, tickers, types_ref, mic_to_exchange):
    hits = []
    unknown_mic_tickers = []
    for symbol, ticker in tickers.items():
        if not ticker.get("eligible"):
            continue  # base universe filter already applied upstream — never recomputed here
        if not eval_preset(preset_cfg, ticker):
            continue
        primary_exchange = (types_ref or {}).get(symbol, {}).get("primary_exchange")
        row, unknown_mic = build_screener_ticker_row(symbol, ticker, primary_exchange, mic_to_exchange)
        hits.append(row)
        if unknown_mic:
            unknown_mic_tickers.append({"symbol": symbol, "primary_exchange": primary_exchange})

    sort_key = SCREENER_SORT_KEYS.get(preset_id)
    if sort_key:
        hits.sort(key=sort_key)
    else:
        hits.sort(key=lambda r: r["symbol"])

    return {
        "name": preset_cfg["name"],
        "count": len(hits),
        "tickers": hits,
    }, unknown_mic_tickers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Dashboard Predefined Screeners Builder")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--market-features", default="data/market_features.json")
    parser.add_argument("--taxonomy-dir", default="data/taxonomy")
    parser.add_argument("--config", default="config/narrative_engine.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 YOLO Dashboard — Predefined Screeners Builder")
    print(f"   Zeit: {datetime.now().isoformat()}")
    print("=" * 60)

    cfg = load_json(args.config)
    if cfg is None:
        print(f"FATAL: {args.config} nicht gefunden.", file=sys.stderr)
        sys.exit(1)
    mic_to_exchange = cfg.get("tradingview_export", {}).get("mic_to_exchange", {})
    presets = cfg.get("screener_presets", {})

    market_features = load_json(args.market_features)
    if market_features is None:
        print(f"FATAL: {args.market_features} nicht gefunden — Screener-Lauf uebersprungen.", file=sys.stderr)
        sys.exit(1)
    tickers = market_features.get("tickers", {})

    types_ref_data = load_json(Path(args.taxonomy_dir) / "market_reference_types.json")
    types_ref = (types_ref_data or {}).get("tickers", {})

    eligible_universe_size = sum(1 for t in tickers.values() if t.get("eligible"))

    screeners = {}
    all_unknown_mic_tickers = {}
    for preset_id, preset_cfg in presets.items():
        if preset_id.startswith("_") or not preset_cfg.get("enabled", False):
            continue
        result, unknown_mic_tickers = build_screener(preset_id, preset_cfg, tickers, types_ref, mic_to_exchange)
        screeners[preset_id] = result
        if unknown_mic_tickers:
            all_unknown_mic_tickers[preset_id] = unknown_mic_tickers
        print(f"  ✅ {result['name']}: {result['count']} Treffer"
              + (f" ({len(unknown_mic_tickers)} mit unbekanntem MIC — siehe meta.unknown_mic_tickers)"
                 if unknown_mic_tickers else ""))

    output = {
        "meta": {
            "as_of_date": (market_features.get("meta") or {}).get("date_range", [None, None])[-1],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "data/market_features.json (Massive, via build_market_features.py)",
            "eligible_universe_size": eligible_universe_size,
            "unknown_mic_tickers": all_unknown_mic_tickers,
        },
        "screeners": screeners,
    }

    out_path = out_dir / "screeners.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n✅ Screeners geschrieben → {out_path} ({size_kb:.0f} KB)")
    print(f"   Eligible Universe: {eligible_universe_size} | Presets: {len(screeners)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
