"""Build Market Overview offline, using canonical QQQ Health formulas on EOD bars."""
import argparse
import json
import math
from datetime import date
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def eod_health(chart, market, breadth, day, config, previous=None):
    """Reject mixed sessions and feed the canonical scorer the chart's own prices."""
    if __package__:
        from .build_dashboard_states import build_qqq_health
    else:
        from build_dashboard_states import build_qqq_health

    bars = chart.get("bars", [])
    if chart.get("symbol") != "QQQ" or len(bars) < 15:
        raise ValueError("QQQ EOD chart needs at least 15 actual candles")
    dates = [bar["date"] for bar in bars]
    if dates != sorted(set(dates)) or dates[-1] != day:
        raise ValueError("QQQ last candle/session order differs from score date")
    for bar in bars:
        if not all(isinstance(bar.get(k), (int, float)) and math.isfinite(bar[k])
                   for k in ("o", "h", "l", "c", "v")):
            raise ValueError("QQQ EOD chart contains invalid OHLCV")
        if not (0 < bar["l"] <= min(bar["o"], bar["c"]) <= max(bar["o"], bar["c"]) <= bar["h"] and bar["v"] > 0):
            raise ValueError("QQQ EOD chart contains invalid OHLCV")
    last = bars[-1]
    benchmark = market.get("QQQ", {})
    for field in ("c", "ema10", "ema20", "sma50", "sma200"):
        value = last.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"QQQ EOD chart missing {field}")
        if not isinstance(benchmark.get(field), (int, float)) or not math.isclose(value, benchmark[field], abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError(f"QQQ benchmark/chart disagree on {field}")
    if benchmark.get("date") != day:
        raise ValueError("QQQ benchmark has a different session date")
    label = date.fromisoformat(day).strftime("%d.%m.")
    # Existing Nasdaq histories carry day/month labels. Validate their actual
    # terminal labels and scalar endpoints, rather than trusting state.meta.
    for values_key, dates_key, scalar in (
        ("mco_history", "mco_zscore_dates", "mco"),
        ("mco_zscore_history", "mco_zscore_dates", "mco_zscore"),
        ("summation_zscore_history", "summation_zscore_dates", "summation_zscore"),
        ("summation_zscore_sma10_history", "summation_zscore_dates", "summation_zscore_sma10"),
        ("hl_history", "hl_dates", "hl_osc"),
    ):
        values, labels = breadth.get(values_key, []), breadth.get(dates_key, [])
        if not values or len(values) != len(labels) or labels[-1] not in (label, day):
            raise ValueError(f"Nasdaq {values_key} does not end on score session")
        if values[-1] != breadth.get(scalar):
            raise ValueError(f"Nasdaq {scalar} differs from history endpoint")
    kma_labels = breadth.get("kma_dates", [])
    for key in ("ema10", "ema21", "sma50", "sma200"):
        values = breadth.get("kma_history", {}).get(key, [])
        if not values or len(values) != len(kma_labels) or kma_labels[-1] not in (label, day) or values[-1] != breadth.get("kma_now", {}).get(key):
            raise ValueError(f"Nasdaq {key} breadth does not match score session")
    # Same simple 14-session True Range mean and rounded ATR% as the legacy
    # calc_atr_extension. No Wilder smoothing, no change to weights/thresholds.
    ranges = [max(current["h"] - current["l"], abs(current["h"] - prior["c"]), abs(current["l"] - prior["c"]))
              for prior, current in zip(bars[-15:-1], bars[-14:])]
    atr = sum(ranges) / 14
    atr_pct = round(atr / last["c"] * 100, 3)
    regime = {"price": last["c"], "mas": {k: last[k] for k in ("ema10", "ema20", "sma50", "sma200")},
              "atr14": round(atr, 4), "atr_pct": atr_pct,
              "atr_extension": round(((last["c"] - last["sma50"]) / last["sma50"] * 100) / atr_pct, 2) if atr_pct > 0 else None}
    health, _ = build_qqq_health(regime, breadth, config["qqq_health_v1"], previous)
    health.update({"as_of_date": day, "price_source": "Massive EOD"})
    return health


def build(data, eod_store=None, config_path=None, history_dir=None):
    state = read(data / "dashboard_state.json")
    features = read(data / "market_features.json")
    narratives = read(data / "narratives.json")
    snapshot = read(data / "snapshot.json")
    day = state["meta"]["as_of_date"]
    for source in (features, narratives):
        if source["meta"]["date_range"][-1] != day:
            raise ValueError("Market Overview requires matching session dates")
    # Legacy snapshot only exposes a generation timestamp, not a session date.
    # Preserve that distinction instead of declaring all inputs EOD synchronized.
    tickers = features["tickers"]
    eligible = [v for v in tickers.values() if v.get("eligible")]

    def above(field):
        values = [v[field] for v in eligible if v.get(field) is not None]
        return round(100 * sum(v > 0 for v in values) / len(values), 1) if values else None

    breadth = snapshot.get("qqq_breadth", {})
    items = state.get("opportunities", {}).get("items", [])
    leadership = sorted((v for v in items if v.get("rs_3m") is not None),
                        key=lambda v: (-v["rs_3m"], v["symbol"]))[:5]
    opportunities = []
    for item in items:
        setups = []
        if item.get("ema10_pullback"):
            setups.append("Pullback an EMA10")
        if item.get("ema20_pullback"):
            setups.append("Pullback an EMA20")
        if item.get("constructive_reset_narratives"):
            setups.append("Constructive Reset")
        if item.get("quality_state") == "fresh_leader":
            setups.append("Fresh Leader")
        if setups and not item.get("extended_v2") and item.get("quality_v2") != "laggard":
            opportunities.append({"symbol": item["symbol"], "setup": " · ".join(setups),
                                  "score": item.get("structural_rs")})
    opportunities.sort(key=lambda v: (-(v["score"] if v["score"] is not None else -1), v["symbol"]))
    history = []
    for path in sorted((data / "history/daily").glob("*.json")):
        if path.stem <= day:
            previous = read(path)
            value = previous.get("market_regime", {}).get("components", {}).get("breadth", {}).get("pct_above_sma50")
            if value is not None:
                history.append({"date": path.stem, "value": value})
    # Compact daily history does not always retain breadth components.
    chart = {"bars": [], "symbol": "QQQ", "as_of_date": day}
    market = {s: {"c": v.get("price"), **v.get("mas", {})}
              for s, v in snapshot.get("regime", {}).items()}
    health = state["qqq_health"]
    if eod_store:
        manifest = read(eod_store / "latest.json")  # Pin one immutable revision.
        if manifest["as_of_date"] != day:
            raise ValueError("EOD snapshot and existing scores have different session dates")

        def pinned(relative):
            path = (eod_store / relative).resolve()
            if not path.is_relative_to(eod_store.resolve()):
                raise ValueError("EOD path escapes store")
            return read(path)

        market_payload = pinned(manifest["market"])
        chart = pinned(manifest["charts"] + "QQQ.json")
        if any(p["as_of_date"] != day for p in (market_payload, chart)):
            raise ValueError("EOD payload date differs from pinned manifest")
        market = market_payload["benchmarks"]
        config_path = config_path or Path(__file__).resolve().parents[1] / "config/narrative_engine.json"
        prior_paths = sorted(p for p in (history_dir or data / "history/daily").glob("*.json") if p.stem < day)
        previous = read(prior_paths[-1]).get("qqq_health") if prior_paths else None
        health = eod_health(chart, market, breadth, day, read(config_path), previous)
    names = {m["symbol"]: m.get("name", m["symbol"])
             for n in narratives["narratives"] for m in n.get("members", [])}
    return {
        "meta": {"as_of_date": day, "snapshot_updated_at": snapshot.get("meta", {}).get("updated_at"),
                 "eod_pinned": bool(eod_store), "session_synchronized": bool(eod_store),
                 "eod_version": manifest["version"] if eod_store else None,
                 "source": "Massive EOD / bestehende Score-Formeln; Nasdaq-Breadth: Yahoo Finance"},
        "regime": state["market_regime"], "health": health,
        "breadth": {"above_sma50": above("sma50_distance_pct"), "above_sma200": above("sma200_distance_pct"),
                    "universe_size": len(eligible), "history": history},
        "nasdaq": {k: breadth.get(k) for k in ("mco", "summation", "new_highs", "new_lows", "mco_history", "mco_zscore_dates", "kma_history", "kma_dates", "n_components")},
        "benchmarks": market, "qqq_chart": chart,
        "narratives": sorted(({"name": n["name"], "score": n.get("narrative_structural_score")}
                              for n in narratives["narratives"] if n.get("narrative_structural_score") is not None),
                             key=lambda n: -n["score"])[:5],
        "leadership": [{"symbol": v["symbol"], "name": names.get(v["symbol"], v["symbol"]), "score": v["rs_3m"]} for v in leadership],
        "opportunities": opportunities[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--eod-store", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--history-dir", type=Path, help="Prior daily states for canonical score delta/hysteresis")
    parser.add_argument("--output", type=Path, default=Path("data/market_overview.json"))
    args = parser.parse_args()
    payload = build(args.data, args.eod_store, args.config, args.history_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Market Overview: {payload['meta']['as_of_date']} -> {args.output}")


if __name__ == "__main__":
    main()
