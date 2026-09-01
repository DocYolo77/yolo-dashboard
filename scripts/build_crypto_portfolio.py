#!/usr/bin/env python3
"""
YOLO Dashboard — Krypto-Portfolio Card

Manually-maintained personal crypto positions (config/crypto_portfolio.json:
entry_date, cash_pct, and per-position allocation_pct/avg_entry -- no
market API can supply these, the user provides them directly and Claude
updates the file in chat) combined with LIVE current prices (yfinance,
same source/library as the public Krypto market table in build_data.py,
fetched independently here so adding a portfolio-only coin like HYPE never
changes what that public table shows).

Weighted performance formulas (verified against the user's own reference
screenshot numbers):
  weighted_position_pct  = sum(allocation_pct/100 * delta_pct) across
                            positions -- allocation_pct weights sum to 100%
                            WITHIN the invested crypto sleeve, never
                            averaged/blended with cash.
  portfolio_impact_pct   = weighted_position_pct * invested_pct/100 -- the
                            same number scaled down by how much of the
                            total account is actually in crypto right now
                            (cash_pct/invested_pct are a separate, directly
                            user-supplied number, not derived from position
                            sizes -- see config's own comment).

Output: data/crypto_portfolio.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf


def load_config(path="config/crypto_portfolio.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_current_price(symbol):
    """Latest valid (non-NaN) close for a single yfinance symbol. Same
    dropna-first NaN-safety pattern as build_data.py's calc_metrics (V6
    point 29A) and fetch_regime_data (2026-08-30 fix) -- a bad trailing
    close must never silently become NaN and break this file's JSON.
    Returns None (never raises) on any fetch/data problem so one bad
    ticker can't fail the whole build."""
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        if hist.empty:
            return None
        close = hist["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception as e:
        print(f"  ⚠ Fehler bei {symbol}: {e}", file=sys.stderr)
        return None


def calc_delta_pct(current_price, avg_entry):
    """% change of current_price vs. avg_entry. None if either input is
    missing/non-positive -- never a fabricated 0%."""
    if current_price is None or not avg_entry:
        return None
    return round((current_price - avg_entry) / avg_entry * 100, 2)


def build_positions(cfg, price_fetcher=fetch_current_price):
    positions = []
    for pos in cfg["positions"]:
        current = price_fetcher(pos["symbol"])
        positions.append({
            "symbol": pos["symbol"],
            "label": pos["label"],
            "allocation_pct": pos["allocation_pct"],
            "avg_entry": pos["avg_entry"],
            "current_price": round(current, 4) if current is not None else None,
            "delta_pct": calc_delta_pct(current, pos["avg_entry"]),
        })
    return positions


def compute_weighted_pct(positions):
    """sum(allocation_pct/100 * delta_pct) over positions with a known
    delta_pct. None if NOT ALL positions have a valid delta_pct -- a
    partial weighted sum would understate/misrepresent the real weighted
    performance (allocation_pct is meant to sum to 100% across ALL
    positions), so this deliberately does not renormalize over a subset."""
    if any(p["delta_pct"] is None for p in positions):
        return None
    return round(sum(p["allocation_pct"] / 100 * p["delta_pct"] for p in positions), 2)


def compute_days_since_entry(entry_date_str, as_of):
    """Inclusive day count (entry day itself counts as day 1), matching how
    the user refers to trade age ("8 Tage" for a Mon 25.8. entry as of a
    following Tue 1.9.)."""
    entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    return (as_of.date() - entry_date).days + 1


def main():
    cfg = load_config()
    positions = build_positions(cfg)
    weighted_pct = compute_weighted_pct(positions)
    cash_pct = cfg["cash_pct"]
    invested_pct = round(100 - cash_pct, 2)
    portfolio_impact_pct = round(weighted_pct * invested_pct / 100, 2) if weighted_pct is not None else None

    now = datetime.now(timezone.utc)
    payload = {
        "meta": {
            "updated_at": now.isoformat(),
            "entry_date": cfg["entry_date"],
            "days_since_entry": compute_days_since_entry(cfg["entry_date"], now),
            "cash_pct": cash_pct,
            "invested_pct": invested_pct,
            "weighted_position_pct": weighted_pct,
            "portfolio_impact_pct": portfolio_impact_pct,
        },
        "positions": positions,
    }

    out_path = Path("data") / "crypto_portfolio.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # allow_nan=False: same reliability fix as everywhere else in this
        # pipeline -- fail loudly here instead of shipping invalid JSON.
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"✅ Krypto-Portfolio geschrieben → {out_path} "
          f"(gewichtete Performance: {weighted_pct}%, Gesamtdepot-Impact: {portfolio_impact_pct}%)")


if __name__ == "__main__":
    main()
