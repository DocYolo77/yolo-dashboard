"""Build a compact Market Overview payload without network calls or score changes."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(data, eod_store=None):
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
    names = {m["symbol"]: m.get("name", m["symbol"])
             for n in narratives["narratives"] for m in n.get("members", [])}
    return {
        "meta": {"as_of_date": day, "snapshot_updated_at": snapshot.get("meta", {}).get("updated_at"),
                 "eod_pinned": bool(eod_store), "source": "Bestehende Dashboard-Scores / Massive; Nasdaq-Breadth: Yahoo Finance"},
        "regime": state["market_regime"], "health": state["qqq_health"],
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
    parser.add_argument("--output", type=Path, default=Path("data/market_overview.json"))
    args = parser.parse_args()
    payload = build(args.data, args.eod_store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Market Overview: {payload['meta']['as_of_date']} -> {args.output}")


if __name__ == "__main__":
    main()
