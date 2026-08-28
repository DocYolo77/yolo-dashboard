#!/usr/bin/env python3
"""
YOLO Dashboard — Market Reference Cache
Maintains two Massive-API-backed reference datasets that change slowly and
are therefore deliberately NOT refreshed on every daily run:

  1. `types`   — bulk ticker list (GET /v3/reference/tickers, cursor-paginated,
                 ~10-30 requests for the whole US stock market). Gives us
                 `type` (CS/ETF/ETN/.../for asset-class exclusion) and
                 `active`/`primary_exchange` at effectively zero marginal
                 cost per ticker. Intended to run weekly.

  2. `enrich`  — per-ticker overview (GET /v3/reference/tickers/{ticker}),
                 the ONLY Massive endpoint that exposes market_cap, SIC and
                 company description. There is no bulk market-cap endpoint
                 (checked against the official docs: /v3/reference/tickers
                 and the grouped-daily/snapshot endpoints do not return
                 market_cap). Calling this for the ~9-11k "type=CS" tickers
                 every day would mean 9-11k HTTP calls in a single GitHub
                 Actions job — a real payload/rate-limit risk, not a
                 hypothetical one. So `enrich` is called ONLY for tickers
                 that already passed the (cheap, market-wide) ADR20 filter
                 in build_market_features.py, and results are cached with a
                 timestamp so a ticker is not re-enriched more than once a
                 week.

Output:
  data/taxonomy/market_reference_types.json   (stage 1, full stock list)
  data/taxonomy/market_reference_cache.json   (stage 2, enriched subset)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

MASSIVE_BASE = "https://api.massive.com"
REQUEST_DELAY_SEC = float(os.environ.get("MASSIVE_REQUEST_DELAY_MS", "150")) / 1000.0
ENRICH_CACHE_MAX_AGE_DAYS = 7
# Reliability fix (2026-08-28): a failed lookup (404 for a permanently
# delisted/renamed ticker, or a transient network/API issue) was previously
# NEVER cached, so build_market_features.py re-attempted the SAME doomed
# request on every single daily run, forever -- observed to help push a
# real run past the "Build full-market features" step's 20-minute timeout
# when Massive's responses were unusually slow that day. A shorter TTL than
# successful entries so a ticker that becomes resolvable again (rare, but
# possible after a re-listing) isn't permanently skipped either.
NEGATIVE_ENRICH_CACHE_MAX_AGE_DAYS = 3


def massive_get(path, params=None, retries=3):
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        print("FATAL: MASSIVE_API_KEY nicht gesetzt.", file=sys.stderr)
        sys.exit(1)
    headers = {"Authorization": f"Bearer {key}"}
    url = path if path.startswith("http") else f"{MASSIVE_BASE}{path}"
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  ⚠ Rate limited, warte {wait}s...", file=sys.stderr)
                time.sleep(wait)
                last_err = "HTTP 429"
                continue
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1)
    print(f"  ⚠ Request fehlgeschlagen ({path}): {last_err}", file=sys.stderr)
    return None


def load_config(path="config/narrative_engine.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# STAGE 1: bulk types list
# ─────────────────────────────────────────────

def fetch_all_stock_tickers():
    """Paginate GET /v3/reference/tickers?market=stocks&active=true across the
    whole US stock market. Returns {ticker: {type, primary_exchange, name}}."""
    print("\n📋 Lade Ticker-Typen-Liste (bulk, /v3/reference/tickers)...")
    out = {}
    params = {"market": "stocks", "active": "true", "limit": 1000}
    url = "/v3/reference/tickers"
    page = 0
    while url:
        data = massive_get(url, params=params if page == 0 else None)
        page += 1
        if not data:
            break
        for row in data.get("results", []):
            sym = row.get("ticker")
            if not sym:
                continue
            out[sym] = {
                "type": row.get("type"),
                "primary_exchange": row.get("primary_exchange"),
                "name": row.get("name"),
                "locale": row.get("locale"),
            }
        print(f"  → Seite {page}: {len(data.get('results', []))} Ticker (kumuliert {len(out)})")
        next_url = data.get("next_url")
        url = next_url
        time.sleep(REQUEST_DELAY_SEC)
    print(f"  ✅ {len(out)} aktive US-Stock-Ticker (alle Typen) geladen")
    return out


def build_types(out_path):
    tickers = fetch_all_stock_tickers()
    payload = {
        "meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Massive /v3/reference/tickers (bulk, market=stocks, active=true)",
            "n_tickers": len(tickers),
        },
        "tickers": tickers,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Typen-Liste geschrieben → {out_path}")
    return payload


# ─────────────────────────────────────────────
# STAGE 2: per-ticker enrichment (market_cap, SIC, description)
# ─────────────────────────────────────────────

def fetch_ticker_overview(ticker):
    data = massive_get(f"/v3/reference/tickers/{ticker}")
    if not data or "results" not in data:
        return None
    r = data["results"]
    return {
        "market_cap": r.get("market_cap"),
        "share_class_shares_outstanding": r.get("share_class_shares_outstanding"),
        "weighted_shares_outstanding": r.get("weighted_shares_outstanding"),
        "sic_code": r.get("sic_code"),
        "sic_description": r.get("sic_description"),
        "description": r.get("description"),
        "total_employees": r.get("total_employees"),
    }


def load_enrich_cache(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("tickers", {})


def enrich_candidates(candidates, cache_path, max_age_days=ENRICH_CACHE_MAX_AGE_DAYS,
                       negative_max_age_days=NEGATIVE_ENRICH_CACHE_MAX_AGE_DAYS, max_calls=None):
    """Enrich only `candidates` (a list of tickers) that are missing from the
    cache or whose cached entry has expired. Merges into the existing cache
    so unrelated tickers already cached are preserved.

    A candidate whose lookup fails (404 for a delisted/renamed ticker, or a
    transient error) is now ALSO cached, as {"not_found": True, "cached_at":
    ...} with its own shorter negative_max_age_days TTL -- previously only a
    successful result was ever cached, so a permanently-dead ticker was
    re-queried on every single run, forever. Downstream consumers read this
    the same as "no entry" (plain dict .get() on missing fields), so no
    other code needed to change."""
    cache = load_enrich_cache(cache_path)
    now = datetime.now(timezone.utc)
    to_fetch = []
    for sym in candidates:
        entry = cache.get(sym)
        if entry is None:
            to_fetch.append(sym)
            continue
        cached_at = entry.get("cached_at")
        try:
            age_days = (now - datetime.fromisoformat(cached_at)).days if cached_at else 999
        except ValueError:
            age_days = 999
        ttl = negative_max_age_days if entry.get("not_found") else max_age_days
        if age_days >= ttl:
            to_fetch.append(sym)

    if max_calls is not None:
        to_fetch = to_fetch[:max_calls]

    print(f"\n💰 Market-Cap/SIC-Enrichment: {len(to_fetch)} von {len(candidates)} Kandidaten benoetigen ein Update "
          f"(Cache-Alter > {max_age_days} Tage bzw. {negative_max_age_days} Tage fuer nicht gefundene Ticker, oder neu)")

    for i, sym in enumerate(to_fetch, 1):
        overview = fetch_ticker_overview(sym)
        if overview is not None:
            overview["cached_at"] = now.isoformat()
            cache[sym] = overview
        else:
            cache[sym] = {"not_found": True, "cached_at": now.isoformat()}
        if i % 100 == 0:
            print(f"  → {i}/{len(to_fetch)} angereichert")
        time.sleep(REQUEST_DELAY_SEC)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "updated_at": now.isoformat(),
                "source": "Massive /v3/reference/tickers/{ticker} (per-ticker, cached)",
                "n_tickers": len(cache),
                "max_age_days": max_age_days,
                "negative_max_age_days": negative_max_age_days,
                "not_found_count": sum(1 for v in cache.values() if v.get("not_found")),
            },
            "tickers": cache,
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ Enrichment-Cache geschrieben → {cache_path} ({len(cache)} Ticker gesamt, {len(to_fetch)} aktualisiert)")
    return cache


def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Market Reference Cache")
    parser.add_argument("stage", choices=["types", "enrich"])
    parser.add_argument("--out-dir", default="data/taxonomy")
    parser.add_argument("--candidates-file", help="JSON file with a list of tickers to enrich (stage=enrich)")
    parser.add_argument("--max-calls", type=int, default=None, help="Safety cap on per-ticker calls this run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.stage == "types":
        build_types(out_dir / "market_reference_types.json")
    else:
        if not args.candidates_file:
            print("FATAL: --candidates-file erforderlich fuer stage=enrich", file=sys.stderr)
            sys.exit(1)
        with open(args.candidates_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        enrich_candidates(candidates, out_dir / "market_reference_cache.json", max_calls=args.max_calls)


if __name__ == "__main__":
    main()
