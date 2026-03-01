#!/usr/bin/env python3
"""
YOLO Dashboard — Data Builder
Fetches market data via yfinance, calculates regime/MAs/breadth,
calls Claude API for AI summaries, outputs data/snapshot.json
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Which briefing to generate: "morning" or "evening"
# Passed via --briefing flag or BRIEFING_TYPE env var

TICKERS = {
    "futures": {
        "ES=F":  "ES (S&P 500)",
        "NQ=F":  "NQ (Nasdaq 100)",
        "YM=F":  "YM (Dow Jones)",
        "RTY=F": "RTY (Russell 2000)",
    },
    "europe": {
        "^GDAXI": "🇩🇪 DAX 40",
        "^FCHI":  "🇫🇷 CAC 40",
        "^FTSE":  "🇬🇧 FTSE 100",
        "^STOXX50E": "🇪🇺 Euro Stoxx 50",
    },
    "global": {
        "^N225":  "🇯🇵 Nikkei 225",
        "^HSI":   "🇭🇰 Hang Seng",
        "000300.SS": "🇨🇳 CSI 300",
        "^AXJO":  "🇦🇺 ASX 200",
    },
    "crypto": {
        "BTC-USD": "Bitcoin (BTC)",
        "ETH-USD": "Ethereum (ETH)",
        "SOL-USD": "Solana (SOL)",
    },
    "commodities": {
        "GC=F":  "Gold (XAU)",
        "SI=F":  "Silber (XAG)",
        "CL=F":  "WTI Crude",
        "BZ=F":  "Brent Crude",
        "NG=F":  "Erdgas",
        "HG=F":  "Kupfer (HG)",
    },
    "currencies": {
        "EURUSD=X": "EU",
        "GBPUSD=X": "GU",
        "DX-Y.NYB": "DXY",
    },
    "vix": {
        "^VIX": "VIX",
    },
    "regime": {
        "SPY": "SPY",
        "QQQ": "QQQ",
    },
    "sectors": {
        "XLK":  "Technologie",
        "XLC":  "Kommunikation",
        "XLI":  "Industrie",
        "XLF":  "Finanzen",
        "XLV":  "Gesundheit",
        "XLY":  "Zyklisch Konsum",
        "XLB":  "Materialien",
        "XLE":  "Energie",
        "XLU":  "Versorger",
        "XLRE": "Immobilien",
        "XLP":  "Basiskons.",
    },
    "themes": {
        "BOTZ": "🤖 KI & Robotik",
        "SKYY": "☁️ Cloud Computing",
        "HACK": "🔒 Cybersecurity",
        "SMH":  "⚡ Halbleiter",
        "XBI":  "🧬 Biotech",
        "ICLN": "🌱 Clean Energy",
        "PAVE": "🏗️ Infrastruktur",
        "ARKG": "💊 Genomik",
        "HERO": "🎮 Gaming & eSport",
        "LIT":  "⛏️ Lithium & Batterie",
    },
    "countries": {
        "ARGT": "🇦🇷 Argentinien",
        "INDA": "🇮🇳 Indien",
        "EWY":  "🇰🇷 Südkorea",
        "EWZ":  "🇧🇷 Brasilien",
        "EWG":  "🇩🇪 Deutschland",
        "EWJ":  "🇯🇵 Japan",
        "EWU":  "🇬🇧 UK",
        "MCHI": "🇨🇳 China",
        "EWQ":  "🇫🇷 Frankreich",
        "EWA":  "🇦🇺 Australien",
        "EWT":  "🇹🇼 Taiwan",
        "EWS":  "🇸🇬 Singapur",
        "THD":  "🇹🇭 Thailand",
        "ECH":  "🇨🇱 Chile",
        "TUR":  "🇹🇷 Türkei",
    },
}

# Breadth proxies (ETFs that track % above MAs etc.)
BREADTH_TICKERS = {
    # These are approximations — real breadth data needs dedicated sources
    # The script uses RSP (equal weight S&P) vs SPY for basic breadth
    "RSP": "S&P 500 Equal Weight",
    "SPY": "S&P 500",
}


def fetch_ticker_data(symbol, period="1y"):
    """Fetch historical data for a single ticker."""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"  ⚠ Error fetching {symbol}: {e}")
        return None


def calc_metrics(hist):
    """Calculate 1D%, 1W%, 52W High%, YTD% from historical data."""
    if hist is None or len(hist) < 2:
        return None

    close = hist["Close"]
    current = close.iloc[-1]

    # 1D%
    prev = close.iloc[-2] if len(close) >= 2 else current
    d1_pct = ((current - prev) / prev) * 100

    # 1W%
    w1_close = close.iloc[-6] if len(close) >= 6 else close.iloc[0]
    w1_pct = ((current - w1_close) / w1_close) * 100

    # 52W High %
    high_52w = close.max()
    hi_pct = ((current - high_52w) / high_52w) * 100

    # YTD%
    year_start = hist[hist.index.year == datetime.now().year]
    if len(year_start) > 0:
        ytd_start = year_start["Close"].iloc[0]
        ytd_pct = ((current - ytd_start) / ytd_start) * 100
    else:
        ytd_pct = 0.0

    return {
        "price": round(current, 2),
        "d1_pct": round(d1_pct, 2),
        "w1_pct": round(w1_pct, 2),
        "hi52w_pct": round(hi_pct, 2),
        "ytd_pct": round(ytd_pct, 2),
    }


def calc_moving_averages(hist):
    """Calculate EMA10, EMA20, SMA50, SMA100, SMA200."""
    if hist is None or len(hist) < 200:
        # Need at least 200 days for SMA200
        if hist is not None and len(hist) >= 50:
            close = hist["Close"]
            result = {}
            for label, n, ema in [("ema10", 10, True), ("ema20", 20, True),
                                   ("sma50", 50, False)]:
                if len(close) >= n:
                    val = close.ewm(span=n).mean().iloc[-1] if ema else close.rolling(n).mean().iloc[-1]
                    result[label] = round(val, 2)
            return result
        return None

    close = hist["Close"]
    return {
        "ema10":  round(close.ewm(span=10).mean().iloc[-1], 2),
        "ema20":  round(close.ewm(span=20).mean().iloc[-1], 2),
        "sma50":  round(close.rolling(50).mean().iloc[-1], 2),
        "sma100": round(close.rolling(100).mean().iloc[-1], 2),
        "sma200": round(close.rolling(200).mean().iloc[-1], 2),
    }


def determine_regime(price, mas):
    """
    BULL: Price > EMA10 and > EMA20
    CHOP: Below EMA10 or EMA20 but above SMA200
    BEAR: Below SMA200
    """
    if mas is None:
        return "UNKNOWN"

    above_ema10 = price > mas.get("ema10", 0)
    above_ema20 = price > mas.get("ema20", 0)
    above_sma200 = price > mas.get("sma200", 0)

    if above_ema10 and above_ema20:
        return "BULL"
    elif above_sma200:
        return "CHOP"
    else:
        return "BEAR"


def fetch_category(category_name, tickers_dict):
    """Fetch data for an entire category of tickers."""
    print(f"\n📊 Fetching {category_name}...")
    results = []
    for symbol, name in tickers_dict.items():
        print(f"  → {symbol} ({name})")
        hist = fetch_ticker_data(symbol)
        metrics = calc_metrics(hist)
        if metrics:
            metrics["symbol"] = symbol
            metrics["name"] = name
            results.append(metrics)
    return results


def fetch_regime_data():
    """Fetch SPY and QQQ with moving averages for regime detection."""
    print("\n🎯 Fetching regime data (SPY/QQQ)...")
    regime = {}
    for symbol in ["SPY", "QQQ"]:
        print(f"  → {symbol}")
        hist = fetch_ticker_data(symbol)
        if hist is not None and len(hist) > 0:
            price = round(hist["Close"].iloc[-1], 2)
            mas = calc_moving_averages(hist)
            r = determine_regime(price, mas)
            regime[symbol] = {
                "price": price,
                "regime": r,
                "mas": mas,
            }
    return regime


def get_vix_zone(vix_val):
    """Classify VIX into zones."""
    if vix_val < 15:
        return "NIEDRIG"
    elif vix_val < 20:
        return "NORMAL"
    elif vix_val < 30:
        return "ERHÖHT"
    else:
        return "HOCH"


def build_top10(all_data):
    """Build Top 10 weekly performance across all categories."""
    combined = []
    cat_map = {
        "sectors": "Sektor",
        "themes": "Thema",
        "countries": "Land",
        "commodities": "Rohstoff",
        "crypto": "Krypto",
    }
    for cat_key, cat_label in cat_map.items():
        if cat_key in all_data:
            for item in all_data[cat_key]:
                combined.append({**item, "category": cat_label})

    # Sort by 1W% descending
    combined.sort(key=lambda x: x.get("w1_pct", -999), reverse=True)
    return combined[:10]


def generate_ai_summary(snapshot, briefing_type="morning"):
    """Call Claude API to generate AI summary in German."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ⚠ No ANTHROPIC_API_KEY found, skipping AI summary")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("  ⚠ anthropic package not installed, skipping AI summary")
        return None

    # Build context from snapshot
    regime_info = json.dumps(snapshot.get("regime", {}), indent=2)
    vix_info = json.dumps(snapshot.get("vix", []), indent=2)
    futures_info = json.dumps(snapshot.get("futures", []), indent=2)
    europe_info = json.dumps(snapshot.get("europe", []), indent=2)
    sectors_info = json.dumps(snapshot.get("sectors", [])[:5], indent=2)  # Top 5
    top10_info = json.dumps(snapshot.get("top10", []), indent=2)
    commodities_info = json.dumps(snapshot.get("commodities", []), indent=2)
    currencies_info = json.dumps(snapshot.get("currencies", []), indent=2)

    now_str = datetime.now().strftime("%d. %b %Y")

    if briefing_type == "morning":
        prompt = f"""Du bist der KI-Analyst für das Yolo Dashboard (@Yolo_Investing).
Schreibe ein prägnantes Morgen-Briefing (07:00 CET) auf Deutsch. Maximal 150 Wörter.

Stil: Direkt, klar, keine Floskeln. Wie ein erfahrener Trader seinem Trading-Buddy schreibt.
Beginne mit "Guten Morgen — hier dein Tagesbriefing."

Inhalt:
1. Overnight/Futures-Lage und was die asiatischen Märkte gemacht haben
2. Wichtige Earnings des Tages (recherchiere die bekanntesten Mega-Cap Earnings dieser Woche)
3. Makro-Kalender (wichtige Wirtschaftsdaten heute)
4. "Auf dem Radar" — 1-2 Dinge die auffallen

Nutze <strong> Tags für Hervorhebungen.

Aktuelle Daten:
Regime: {regime_info}
Futures: {futures_info}
Europa: {europe_info}
VIX: {vix_info}
Rohstoffe: {commodities_info}
Währungen: {currencies_info}
Datum: {now_str}"""
    else:
        prompt = f"""Du bist der KI-Analyst für das Yolo Dashboard (@Yolo_Investing).
Schreibe eine prägnante Tages-Zusammenfassung (22:00 CET) auf Deutsch. Maximal 180 Wörter.

Stil: Direkt, klar, analytisch. Wie ein erfahrener Trader den Tag zusammenfasst.
Beginne mit einer fetten Headline die den Tag zusammenfasst.

Inhalt:
1. Tages-Performance und Regime-Status
2. Marktbreite / Breadth
3. Earnings-Highlights des Tages (wichtige Ergebnisse von Mega-Caps)
4. Sektorrotation — wer gewinnt, wer verliert
5. Rohstoffe & Währungen
6. "Warnsignal" oder "Auf dem Radar" — etwas das man beobachten sollte

Nutze <strong> Tags für Hervorhebungen.

Aktuelle Daten:
Regime: {regime_info}
Futures: {futures_info}
Europa: {europe_info}
Top Sektoren: {sectors_info}
Top 10 der Woche: {top10_info}
VIX: {vix_info}
Rohstoffe: {commodities_info}
Währungen: {currencies_info}
Datum: {now_str}"""

    try:
        print(f"  🤖 Calling Claude API for {briefing_type} briefing...")
        message = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        print(f"  ✅ AI summary generated ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"  ⚠ Claude API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Data Builder")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--briefing", default=os.environ.get("BRIEFING_TYPE", "morning"),
                        choices=["morning", "evening"], help="Which AI briefing to generate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 YOLO Dashboard — Data Builder")
    print(f"   Briefing type: {args.briefing}")
    print(f"   Output: {out_dir}")
    print(f"   Time: {datetime.now().isoformat()}")
    print("=" * 60)

    snapshot = {}

    # 1. Regime
    snapshot["regime"] = fetch_regime_data()

    # 2. All categories
    for cat_key, cat_tickers in TICKERS.items():
        if cat_key == "regime":
            continue  # Already handled
        snapshot[cat_key] = fetch_category(cat_key, cat_tickers)

    # 3. Sort sectors, themes, countries by 1W%
    for key in ["sectors", "themes", "countries"]:
        if key in snapshot:
            snapshot[key].sort(key=lambda x: x.get("w1_pct", -999), reverse=True)

    # 4. Countries: keep only top 10
    if "countries" in snapshot:
        snapshot["countries"] = snapshot["countries"][:10]

    # 5. VIX zone
    if snapshot.get("vix") and len(snapshot["vix"]) > 0:
        vix_val = snapshot["vix"][0].get("price", 0)
        snapshot["vix"][0]["zone"] = get_vix_zone(vix_val)

    # 6. Top 10 weekly
    snapshot["top10"] = build_top10(snapshot)

    # 7. AI Summary (optional — works without API key)
    # Load existing snapshot to preserve the other briefing
    existing = {}
    snapshot_path = out_dir / "snapshot.json"
    if snapshot_path.exists():
        try:
            with open(snapshot_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    # Try Claude API first, fall back to manual briefings.json
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    ai_text = None

    if api_key:
        ai_text = generate_ai_summary(snapshot, args.briefing)
    else:
        print("  ℹ Kein ANTHROPIC_API_KEY — nutze manuelle Briefings aus data/briefings.json")
        # Load manual briefings
        briefings_path = out_dir / "briefings.json"
        if briefings_path.exists():
            try:
                with open(briefings_path, encoding="utf-8") as f:
                    manual = json.load(f)
                key = args.briefing  # "morning" or "evening"
                if key in manual:
                    ai_text = manual[key].get("text")
                    print(f"  ✅ Manuelles {key}-Briefing geladen")
            except Exception as e:
                print(f"  ⚠ Fehler beim Lesen von briefings.json: {e}")

    now_str = datetime.now().strftime("%d. %b %Y")

    if args.briefing == "morning":
        snapshot["ai_morning"] = {
            "text": ai_text or "Noch kein Briefing eingetragen. Editiere data/briefings.json auf GitHub.",
            "timestamp": f"{now_str} · 07:00 CET",
        }
        if "ai_evening" in existing:
            snapshot["ai_evening"] = existing["ai_evening"]
    else:
        snapshot["ai_evening"] = {
            "text": ai_text or "Noch keine Zusammenfassung eingetragen. Editiere data/briefings.json auf GitHub.",
            "timestamp": f"{now_str} · 22:00 CET",
        }
        if "ai_morning" in existing:
            snapshot["ai_morning"] = existing["ai_morning"]

    # 8. Metadata
    snapshot["meta"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "briefing_type": args.briefing,
        "source": "Yahoo Finance + Claude AI",
    }

    # 9. Write output
    with open(out_dir / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Snapshot written to {out_dir / 'snapshot.json'}")
    print(f"   Categories: {len(snapshot) - 1}")
    print(f"   AI briefing: {args.briefing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
