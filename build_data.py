#!/usr/bin/env python3
"""
YOLO Dashboard — Data Builder v2
Fetches market data via yfinance, calculates regime/MAs/breadth,
scrapes CNN Fear & Greed, computes McClellan, calls Claude API (optional).
Outputs data/snapshot.json
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import requests

# ─────────────────────────────────────────────
# TICKER CONFIGURATION
# ─────────────────────────────────────────────

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
        "REMX": "🪨 Seltene Erden",
        "SLV":  "🥈 Silber (iShares)",
        "GDXJ": "⛏️ Jr. Goldminen",
        "GDX":  "🥇 Goldminen",
        "COPX": "🔶 Kupferminen",
        "NANR": "🌍 Nat. Ressourcen",
        "OUNZ": "🥇 Gold (VanEck)",
        "SPPP": "⚪ Platin & Palladium",
        "FFTY": "📈 IBD 50 Momentum",
        "IDNA": "🧬 Genomik & Immuno",
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
    "watchlist": {
        "ASM":   "ASM International",
        "CCJ":   "Cameco",
        "HL":    "Hecla Mining",
        "RRX":   "Roper Technologies",
        "SNDK":  "Western Digital",
    },
}


# ─────────────────────────────────────────────
# S&P 500 BREADTH (% above SMAs, A/D, New Hi/Lo)
# ─────────────────────────────────────────────

def get_sp500_tickers():
    """Get S&P 500 ticker list from Wikipedia."""
    try:
        print("  → Lade S&P 500 Komponentenliste...")
        import re
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, timeout=15)
        tickers = []
        rows = resp.text.split('<tr>')
        for row in rows[2:]:
            cells = row.split('<td>')
            if len(cells) > 1:
                match = re.search(r'>([A-Z]{1,5})</a>', cells[1])
                if match:
                    tickers.append(match.group(1).replace('.', '-'))
        if len(tickers) > 400:
            print(f"  ✅ {len(tickers)} Ticker geladen")
            return tickers
        return None
    except Exception as e:
        print(f"  ⚠ Ticker-Liste Fehler: {e}")
        return None


def fetch_breadth_data():
    """Calculate S&P 500 breadth: % above SMA 20/50/200, A/D, New Highs/Lows."""
    import pandas as pd
    print("\n📊 Berechne S&P 500 Breadth...")

    tickers = get_sp500_tickers()
    if not tickers:
        return None

    try:
        # Download in batches of 50 to avoid timeouts
        batch_size = 50
        all_frames = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tickers) + batch_size - 1) // batch_size
            print(f"  → Batch {batch_num}/{total_batches} ({len(batch)} Ticker)...")
            try:
                raw = yf.download(batch, period="14mo", progress=False, threads=True)
                if raw.empty:
                    continue

                # Handle MultiIndex: yfinance returns (Price, Ticker) columns
                if isinstance(raw.columns, pd.MultiIndex):
                    close_df = raw["Close"]
                elif "Close" in raw.columns:
                    # Single ticker case
                    close_df = raw[["Close"]]
                    close_df.columns = [batch[0]]
                else:
                    continue

                all_frames.append(close_df)
            except Exception as e:
                print(f"    ⚠ Batch {batch_num} Fehler: {e}")
                continue

        if not all_frames:
            print("  ⚠ Keine Daten geladen")
            return None

        # Combine all batches
        combined = pd.concat(all_frames, axis=1)
        valid = combined.dropna(axis=1, thresh=200)
        n = len(valid.columns)
        print(f"  → {n} Aktien mit genug Daten")

        if n < 250:
            print(f"  ⚠ Nur {n} Aktien — zu wenig (min. 250)")
            return None

        latest = valid.iloc[-1]
        prev = valid.iloc[-2]

        # % above SMAs
        sma20 = valid.rolling(20).mean().iloc[-1]
        sma50 = valid.rolling(50).mean().iloc[-1]
        sma200 = valid.rolling(200).mean().iloc[-1]

        pct_20 = round(float((latest > sma20).sum()) / n * 100, 1)
        pct_50 = round(float((latest > sma50).sum()) / n * 100, 1)
        pct_200 = round(float((latest > sma200).sum()) / n * 100, 1)

        # Advance/Decline
        change = latest - prev
        adv = int((change > 0).sum())
        dec = int((change < 0).sum())
        ad_ratio = round(adv / max(dec, 1), 2)

        # New 52W Highs/Lows
        hi52 = valid.rolling(252, min_periods=200).max().iloc[-1]
        lo52 = valid.rolling(252, min_periods=200).min().iloc[-1]
        new_hi = int((latest >= hi52 * 0.995).sum())
        new_lo = int((latest <= lo52 * 1.005).sum())

        print(f"  ✅ SMA20: {pct_20}% | SMA50: {pct_50}% | SMA200: {pct_200}%")
        print(f"  ✅ A/D: {adv}/{dec} = {ad_ratio} | Hochs/Tiefs: {new_hi}/{new_lo}")

        return {
            "pct_above_sma20": pct_20,
            "pct_above_sma50": pct_50,
            "pct_above_sma200": pct_200,
            "advance_decline_ratio": ad_ratio,
            "advancers": adv,
            "decliners": dec,
            "new_highs": new_hi,
            "new_lows": new_lo,
            "n_components": n,
        }
    except Exception as e:
        print(f"  ⚠ Breadth Fehler: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────
# CNN FEAR & GREED INDEX
# ─────────────────────────────────────────────

def fetch_fear_greed():
    """Fetch CNN Fear & Greed Index from their API."""
    print("\n😱 Lade CNN Fear & Greed Index...")
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            score = data.get("fear_and_greed", {}).get("score", None)
            rating = data.get("fear_and_greed", {}).get("rating", "")

            if score is not None:
                score = round(score)
                # Translate rating to German
                rating_map = {
                    "Extreme Fear": "Extreme Angst",
                    "Fear": "Angst",
                    "Neutral": "Neutral",
                    "Greed": "Gier",
                    "Extreme Greed": "Extreme Gier",
                }
                rating_de = rating_map.get(rating, rating)
                print(f"  ✅ Fear & Greed: {score} ({rating_de})")
                return {"score": score, "rating": rating_de}

        print(f"  ⚠ Fear & Greed API Status: {resp.status_code}")
        return None
    except Exception as e:
        print(f"  ⚠ Fear & Greed Fehler: {e}")
        return None


# ─────────────────────────────────────────────
# PUT/CALL RATIO
# ─────────────────────────────────────────────

def fetch_put_call():
    """Fetch CBOE Equity Put/Call ratio from their public CSV."""
    print("\n📞 Lade Put/Call Ratio (CBOE)...")
    try:
        # CBOE publishes historical equity put/call as CSV
        url = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            # Find last data line (skip headers and blank lines)
            for line in reversed(lines):
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        ratio = float(parts[4])  # P/C Ratio is the 5th column
                        date = parts[0]
                        print(f"  ✅ CBOE Equity Put/Call: {ratio} (Datum: {date})")
                        return ratio
                    except (ValueError, IndexError):
                        continue

        # Fallback: total put/call ratio
        url2 = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"
        resp2 = requests.get(url2, headers=headers, timeout=15)
        if resp2.status_code == 200:
            lines = resp2.text.strip().split('\n')
            for line in reversed(lines):
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        ratio = float(parts[4])
                        print(f"  ✅ CBOE Total Put/Call: {ratio} (Fallback)")
                        return ratio
                    except (ValueError, IndexError):
                        continue

        print("  ⚠ CBOE CSV nicht verfügbar")
        return None
    except Exception as e:
        print(f"  ⚠ Put/Call Fehler: {e}")
        return None


# ─────────────────────────────────────────────
# CORE MARKET DATA FUNCTIONS
# ─────────────────────────────────────────────

def fetch_ticker_data(symbol, period="1y"):
    """Fetch historical data for a single ticker."""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"  ⚠ Fehler bei {symbol}: {e}")
        return None


def calc_metrics(hist):
    """Calculate 1D%, 1W%, 52W High%, YTD%, hist_5d from historical data."""
    if hist is None or len(hist) < 2:
        return None

    close = hist["Close"]
    current = close.iloc[-1]

    prev = close.iloc[-2] if len(close) >= 2 else current
    d1_pct = ((current - prev) / prev) * 100

    w1_close = close.iloc[-6] if len(close) >= 6 else close.iloc[0]
    w1_pct = ((current - w1_close) / w1_close) * 100

    high_52w = close.max()
    hi_pct = ((current - high_52w) / high_52w) * 100

    year_start = hist[hist.index.year == datetime.now().year]
    if len(year_start) > 0:
        ytd_start = year_start["Close"].iloc[0]
        ytd_pct = ((current - ytd_start) / ytd_start) * 100
    else:
        ytd_pct = 0.0

    # Last 5 trading days for sparkline
    hist_5d = close.iloc[-5:].tolist() if len(close) >= 5 else close.tolist()
    hist_5d = [round(float(x), 2) for x in hist_5d]

    return {
        "price": round(current, 2),
        "d1_pct": round(d1_pct, 2),
        "w1_pct": round(w1_pct, 2),
        "hi52w_pct": round(hi_pct, 2),
        "ytd_pct": round(ytd_pct, 2),
        "hist_5d": hist_5d,
    }


def calc_moving_averages(hist):
    """Calculate EMA10, EMA20, SMA50, SMA100, SMA200."""
    if hist is None or len(hist) < 50:
        return None

    close = hist["Close"]
    result = {}

    for label, n, is_ema in [("ema10", 10, True), ("ema20", 20, True),
                              ("sma50", 50, False), ("sma100", 100, False),
                              ("sma200", 200, False)]:
        if len(close) >= n:
            val = close.ewm(span=n).mean().iloc[-1] if is_ema else close.rolling(n).mean().iloc[-1]
            result[label] = round(val, 2)

    return result if result else None


def determine_regime(price, mas):
    """BULL / CHOP / BEAR based on MA positions."""
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
    print(f"\n📊 Lade {category_name}...")
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
    print("\n🎯 Lade Regime-Daten (SPY/QQQ)...")
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

    combined.sort(key=lambda x: x.get("w1_pct", -999), reverse=True)
    return combined[:10]


# ─────────────────────────────────────────────
# AI SUMMARY (optional — requires ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────

def generate_ai_summary(snapshot, briefing_type="morning"):
    """Call Claude API to generate AI summary in German."""
    try:
        import anthropic
    except ImportError:
        print("  ⚠ anthropic Paket nicht installiert")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    regime_info = json.dumps(snapshot.get("regime", {}), indent=2)
    vix_info = json.dumps(snapshot.get("vix", []), indent=2)
    futures_info = json.dumps(snapshot.get("futures", []), indent=2)
    europe_info = json.dumps(snapshot.get("europe", []), indent=2)
    sectors_info = json.dumps(snapshot.get("sectors", [])[:5], indent=2)
    top10_info = json.dumps(snapshot.get("top10", []), indent=2)
    commodities_info = json.dumps(snapshot.get("commodities", []), indent=2)
    currencies_info = json.dumps(snapshot.get("currencies", []), indent=2)
    fg_info = json.dumps(snapshot.get("fear_greed", {}), indent=2)
    breadth_info = json.dumps(snapshot.get("breadth", {}), indent=2)

    now_str = datetime.now().strftime("%d. %b %Y")

    if briefing_type == "morning":
        prompt = f"""Du bist der KI-Analyst für das Yolo Dashboard (@Yolo_Investing).
Schreibe ein prägnantes Morgen-Briefing (07:00 CET) auf Deutsch. Maximal 150 Wörter.

Stil: Direkt, klar, keine Floskeln. Wie ein erfahrener Trader seinem Trading-Buddy schreibt.
Beginne mit "Guten Morgen — hier dein Tagesbriefing."

Inhalt:
1. Overnight/Futures-Lage und asiatische Märkte
2. Wichtige Earnings des Tages (recherchiere Mega-Cap Earnings dieser Woche)
3. Makro-Kalender (wichtige Wirtschaftsdaten heute)
4. "Auf dem Radar" — 1-2 auffällige Dinge

Nutze <strong> Tags für Hervorhebungen.

Daten:
Regime: {regime_info}
Futures: {futures_info}
Europa: {europe_info}
VIX: {vix_info}
Fear & Greed: {fg_info}
Rohstoffe: {commodities_info}
Währungen: {currencies_info}
Breadth: {breadth_info}
Datum: {now_str}"""
    else:
        prompt = f"""Du bist der KI-Analyst für das Yolo Dashboard (@Yolo_Investing).
Schreibe eine prägnante Tages-Zusammenfassung (22:00 CET) auf Deutsch. Maximal 180 Wörter.

Stil: Direkt, klar, analytisch. Wie ein erfahrener Trader den Tag zusammenfasst.
Beginne mit einer fetten Headline die den Tag zusammenfasst.

Inhalt:
1. Tages-Performance und Regime-Status
2. Marktbreite / Breadth (nutze die echten Zahlen)
3. Earnings-Highlights des Tages
4. Sektorrotation — Gewinner und Verlierer
5. Rohstoffe & Währungen
6. "Warnsignal" oder "Auf dem Radar"

Nutze <strong> Tags für Hervorhebungen.

Daten:
Regime: {regime_info}
Futures: {futures_info}
Europa: {europe_info}
Top Sektoren: {sectors_info}
Top 10 Woche: {top10_info}
VIX: {vix_info}
Fear & Greed: {fg_info}
Rohstoffe: {commodities_info}
Währungen: {currencies_info}
Breadth: {breadth_info}
Datum: {now_str}"""

    try:
        print(f"  🤖 Claude API → {briefing_type}-Briefing...")
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        print(f"  ✅ AI-Briefing generiert ({len(text)} Zeichen)")
        return text
    except Exception as e:
        print(f"  ⚠ Claude API Fehler: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Data Builder")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--briefing", default=os.environ.get("BRIEFING_TYPE", "morning"),
                        choices=["morning", "evening"], help="Which AI briefing to generate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 YOLO Dashboard — Data Builder v2")
    print(f"   Briefing: {args.briefing}")
    print(f"   Output: {out_dir}")
    print(f"   Zeit: {datetime.now().isoformat()}")
    print("=" * 60)

    snapshot = {}

    # 1. Regime
    snapshot["regime"] = fetch_regime_data()

    # 2. All categories
    for cat_key, cat_tickers in TICKERS.items():
        if cat_key == "regime":
            continue
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

    # 7. S&P 500 Breadth
    breadth = fetch_breadth_data()
    if breadth:
        snapshot["breadth"] = breadth

    # 8. CNN Fear & Greed
    fg = fetch_fear_greed()
    if fg:
        snapshot["fear_greed"] = fg

    # 9. Put/Call Ratio (CBOE)
    pc = fetch_put_call()
    if pc:
        snapshot["put_call"] = pc

    # ═══ AI Summary ═══
    # 10. Load existing to preserve the other briefing
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
        print("\n  ℹ Kein ANTHROPIC_API_KEY — nutze manuelle Briefings")
        briefings_path = out_dir / "briefings.json"
        if briefings_path.exists():
            try:
                with open(briefings_path, encoding="utf-8") as f:
                    manual = json.load(f)
                key = args.briefing
                if key in manual:
                    ai_text = manual[key].get("text")
                    print(f"  ✅ Manuelles {key}-Briefing geladen")
            except Exception as e:
                print(f"  ⚠ Fehler bei briefings.json: {e}")

    now_str = datetime.now().strftime("%d. %b %Y")

    if args.briefing == "morning":
        snapshot["ai_morning"] = {
            "text": ai_text or "Noch kein Briefing eingetragen.",
            "timestamp": f"{now_str} · 07:00 CET",
        }
        if "ai_evening" in existing:
            snapshot["ai_evening"] = existing["ai_evening"]
    else:
        snapshot["ai_evening"] = {
            "text": ai_text or "Noch keine Zusammenfassung eingetragen.",
            "timestamp": f"{now_str} · 22:00 CET",
        }
        if "ai_morning" in existing:
            snapshot["ai_morning"] = existing["ai_morning"]

    # 11. Metadata
    snapshot["meta"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "briefing_type": args.briefing,
        "source": "Yahoo Finance + CNN + McClellan berechnet",
    }

    # 12. Write output
    with open(out_dir / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Snapshot geschrieben → {out_dir / 'snapshot.json'}")
    print(f"   Kategorien: {len(snapshot) - 1}")
    if breadth:
        print(f"   Breadth: {breadth['n_components']} Aktien | SMA200: {breadth['pct_above_sma200']}%")
    if fg:
        print(f"   Fear & Greed: {fg['score']} ({fg['rating']})")
    if pc:
        print(f"   Put/Call: {pc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
