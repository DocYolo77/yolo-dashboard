#!/usr/bin/env python3
"""
YOLO Dashboard — Data Builder v5
Fetches market data via yfinance: SPY/QQQ regime (MAs + ATR/ATR-Extension,
feeds QQQ Health), index/crypto/commodity tables, VIX, CNN Fear & Greed,
McClellan Oscillator/Summation Index and % above key moving averages for
NDX 100 (feeds QQQ Health breadth).
Outputs data/snapshot.json

V1 dashboard rebuild changes:
- Waehrungssektion (EUR/USD, GBP/USD, DXY) entfernt — VIX zieht in die neue
  Momentum-Market-Regime-Sektion um, ist dort aber weiterhin nur Kontext,
  kein gewichteter Score-Bestandteil (siehe scripts/build_dashboard_states.py).
- Die alte S&P-500-Breadth-Berechnung (get_sp500_tickers/fetch_breadth_data,
  Wikipedia-Scrape + ~100-500-Ticker-yfinance-Download) wurde entfernt: sie
  wurde im Frontend nie konsumiert und ist durch den neuen, eligible-
  Universe-basierten Market Breadth Score (aus data/market_features.json,
  siehe build_dashboard_states.py) vollstaendig ersetzt. Spart taeglich
  einen teuren Fetch.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import requests

sys.path.insert(0, str(Path(__file__).parent))
from build_market_features import calc_true_range  # noqa: E402  (reuse canonical ATR formula, one definition repo-wide)

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
    "vix": {
        "^VIX": "VIX",
    },
    "regime": {
        "SPY": "SPY",
        "QQQ": "QQQ",
    },
}


# ─────────────────────────────────────────────
# QQQ (NDX 100) McClellan + Summation + H/L Oscillator
# ─────────────────────────────────────────────

NDX100_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    "NFLX","TMUS","ASML","PEP","CSCO","LIN","ADBE","AMD","ISRG","QCOM",
    "TXN","INTU","BKNG","CMCSA","AMGN","HON","AMAT","PANW","ADP","VRTX",
    "GILD","ADI","MU","LRCX","SBUX","MELI","KLAC","REGN","INTC","CDNS",
    "PYPL","CRWD","SNPS","CTAS","MAR","ORLY","MDLZ","CEG","ABNB","FTNT",
    "DASH","CSX","MNST","ADSK","WDAY","PCAR","ROP","CHTR","NXPI","AEP",
    "PAYX","ROST","FANG","KDP","ODFL","FAST","BKR","KHC","EA","DDOG",
    "VRSK","EXC","CTSH","XEL","GEHC","TTWO","CCEP","CSGP","AZN","TEAM",
    "IDXX","ANSS","ZS","ON","CDW","BIIB","DXCM","WBD","MDB","TTD",
    "ARM","MRVL","PLTR","APP","AXON","LULU","MSTR","SMCI","GFS","ILMN",
]


def fetch_qqq_breadth():
    """Calculate QQQ (NDX 100) McClellan Oscillator, Summation Index, H/L Oscillator with history."""
    import pandas as pd
    import numpy as np
    print("\n📈 Berechne QQQ Breadth (McClellan + Summation + H/L)...")

    try:
        # Download all NDX 100 in two batches
        all_frames = []
        for i in range(0, len(NDX100_TICKERS), 50):
            batch = NDX100_TICKERS[i:i + 50]
            print(f"  → Batch {i // 50 + 1}/2 ({len(batch)} Ticker)...")
            try:
                raw = yf.download(batch, period="14mo", progress=False, threads=True)
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    if "Close" in raw.columns.get_level_values(0):
                        close_df = raw["Close"]
                    elif "Close" in raw.columns.get_level_values(1):
                        close_df = raw.xs("Close", level=1, axis=1)
                    else:
                        continue
                elif "Close" in raw.columns:
                    close_df = raw[["Close"]]
                    close_df.columns = [batch[0]]
                else:
                    continue
                all_frames.append(close_df)
            except Exception as e:
                print(f"    ⚠ Batch Fehler: {e}")
                continue

        if not all_frames:
            return None

        combined = pd.concat(all_frames, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        valid = combined.dropna(axis=1, thresh=200)
        n = len(valid.columns)
        print(f"  → {n} QQQ-Komponenten")

        if n < 60:
            print(f"  ⚠ Nur {n} Aktien — zu wenig für QQQ Breadth")
            return None

        # Daily change → advances/declines per day
        change = valid.diff()
        adv_daily = (change > 0).sum(axis=1)
        dec_daily = (change < 0).sum(axis=1)

        # Ratio-Adjusted Net Advances (RANA)
        total = adv_daily + dec_daily
        rana = ((adv_daily - dec_daily) / total.replace(0, 1)) * 1000

        # McClellan Oscillator = 19-day EMA(RANA) - 39-day EMA(RANA)
        ema19 = rana.ewm(span=19, adjust=False).mean()
        ema39 = rana.ewm(span=39, adjust=False).mean()
        mco = ema19 - ema39

        # Normalized MCO (Z-Score) — 200-day rolling mean & std
        mco_clean = mco.dropna()
        mco_mean = mco_clean.rolling(200, min_periods=80).mean()
        mco_std = mco_clean.rolling(200, min_periods=80).std()
        mco_zscore = (mco_clean - mco_mean) / mco_std

        # McClellan Summation Index = cumulative sum of MCO
        summation = mco.cumsum()
        summation = summation - summation.iloc[40]  # offset to ignore initial EMA warmup

        # Normalized MCSI (Z-Score) — same 200-day rolling window as MCO
        sum_clean = summation.dropna()
        sum_mean = sum_clean.rolling(200, min_periods=80).mean()
        sum_std = sum_clean.rolling(200, min_periods=80).std()
        sum_zscore = (sum_clean - sum_mean) / sum_std
        # 10-period SMA of the normalized MCSI (signal line, like TradingView)
        sum_z_sma10 = sum_zscore.rolling(10).mean()

        # Raw EMAs kept for reference
        summation_ema5 = summation.ewm(span=5, adjust=False).mean()
        summation_ema10 = summation.ewm(span=10, adjust=False).mean()

        # Daily H/L Oscillator (new 20-day highs - new 20-day lows per day)
        rolling_hi = valid.rolling(20).max()
        rolling_lo = valid.rolling(20).min()
        new_hi_daily = (valid >= rolling_hi).sum(axis=1)
        new_lo_daily = (valid <= rolling_lo).sum(axis=1)
        hl_osc = new_hi_daily - new_lo_daily

        # % of stocks above KEY moving averages (time series)
        kma_defs = {
            "sma5":   valid.rolling(5).mean(),
            "ema10":  valid.ewm(span=10, adjust=False).mean(),
            "sma20":  valid.rolling(20).mean(),
            "ema21":  valid.ewm(span=21, adjust=False).mean(),
            "sma50":  valid.rolling(50).mean(),
            "sma200": valid.rolling(200).mean(),
        }
        kma_series = {}
        for label, ma in kma_defs.items():
            kma_series[label] = ((valid > ma).sum(axis=1) / n * 100).dropna()

        # Keep last 120 trading days for charts
        history_days = 120
        mco_hist = mco.dropna().iloc[-history_days:]
        mco_z_hist = mco_zscore.dropna().iloc[-history_days:]
        sum_hist = summation.dropna().iloc[-history_days:]
        sum_z_hist = sum_zscore.dropna().iloc[-history_days:]
        sum_z_sma10_hist = sum_z_sma10.dropna().iloc[-history_days:]
        sum_ema5_hist = summation_ema5.dropna().iloc[-history_days:]
        sum_ema10_hist = summation_ema10.dropna().iloc[-history_days:]
        hl_hist = hl_osc.dropna().iloc[-history_days:]

        # Current values
        mco_current = round(float(mco_hist.iloc[-1]), 2)
        mco_z_current = round(float(mco_z_hist.iloc[-1]), 2) if len(mco_z_hist) else 0.0
        sum_current = round(float(sum_hist.iloc[-1]), 1)
        sum_z_current = round(float(sum_z_hist.iloc[-1]), 2) if len(sum_z_hist) else 0.0
        sum_z_sma10_current = round(float(sum_z_sma10_hist.iloc[-1]), 2) if len(sum_z_sma10_hist) else 0.0
        sum_ema5_current = round(float(sum_ema5_hist.iloc[-1]), 1) if len(sum_ema5_hist) else 0.0
        sum_ema10_current = round(float(sum_ema10_hist.iloc[-1]), 1) if len(sum_ema10_hist) else 0.0
        hl_current = int(hl_hist.iloc[-1])
        new_hi_now = int(new_hi_daily.iloc[-1])
        new_lo_now = int(new_lo_daily.iloc[-1])

        # Format history for JSON
        def fmt_hist(s, decimals=2):
            return [round(float(x), decimals) for x in s.tolist()]

        def fmt_dates(s):
            """German short date labels (DD.MM.) for chart x-axes, aligned
            1:1 with the corresponding _history array (same index)."""
            return [d.strftime("%d.%m.") for d in s.index]

        # Build KMA current + history payload
        kma_now = {}
        kma_hist = {}
        for label, series in kma_series.items():
            s = series.iloc[-history_days:]
            kma_now[label] = round(float(s.iloc[-1]), 1) if len(s) else 0.0
            kma_hist[label] = fmt_hist(s, 1)

        print(f"  ✅ MCO: {mco_current} ({mco_z_current:+.2f}σ) | MCSI: {sum_current} ({sum_z_current:+.2f}σ) | H/L: {hl_current} ({new_hi_now}H/{new_lo_now}L)")
        print(f"  ✅ %>MA — 5SMA: {kma_now.get('sma5')}% | 10EMA: {kma_now.get('ema10')}% | 21EMA: {kma_now.get('ema21')}% | 50SMA: {kma_now.get('sma50')}% | 200SMA: {kma_now.get('sma200')}%")

        return {
            "mco": mco_current,
            "mco_zscore": mco_z_current,
            "summation": sum_current,
            "summation_zscore": sum_z_current,
            "summation_zscore_sma10": sum_z_sma10_current,
            "summation_ema5": sum_ema5_current,
            "summation_ema10": sum_ema10_current,
            "hl_osc": hl_current,
            "new_highs": new_hi_now,
            "new_lows": new_lo_now,
            "pct_above_sma20_ndx": kma_now.get("sma20", 0.0),
            "kma_now": kma_now,
            "mco_history": fmt_hist(mco_hist, 2),
            "mco_zscore_history": fmt_hist(mco_z_hist, 2),
            "mco_zscore_dates": fmt_dates(mco_z_hist),
            "summation_history": fmt_hist(sum_hist, 1),
            "summation_zscore_history": fmt_hist(sum_z_hist, 2),
            "summation_zscore_dates": fmt_dates(sum_z_hist),
            "summation_zscore_sma10_history": fmt_hist(sum_z_sma10_hist, 2),
            "summation_ema5_history": fmt_hist(sum_ema5_hist, 1),
            "summation_ema10_history": fmt_hist(sum_ema10_hist, 1),
            "hl_history": [int(x) for x in hl_hist.tolist()],
            "hl_dates": fmt_dates(hl_hist),
            "pct_above_sma20_ndx_history": kma_hist.get("sma20", []),
            "kma_history": kma_hist,
            "kma_dates": fmt_dates(kma_series["sma20"].iloc[-history_days:]),
            "n_components": n,
        }
    except Exception as e:
        print(f"  ⚠ QQQ Breadth Fehler: {e}")
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
    """Fetch live Put/Call ratio from CNN Fear & Greed API (put_call_options component)."""
    print("\n📞 Lade Put/Call Ratio (CNN)...")
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # CNN provides 5-day average put/call ratio via fear & greed components
            pc_data = data.get("put_call_options", {})
            if pc_data:
                # Latest value from time series
                series = pc_data.get("data", [])
                if series:
                    latest = series[-1]
                    ratio = latest.get("y") or latest.get("x")
                    if ratio is not None:
                        ratio = round(float(ratio), 2)
                        ts = latest.get("x", "")
                        # Parse timestamp if present
                        try:
                            from datetime import datetime as dt
                            if isinstance(ts, (int, float)):
                                date_str = dt.fromtimestamp(ts / 1000).strftime("%d.%m.%Y")
                            else:
                                date_str = "live"
                        except Exception:
                            date_str = "live"
                        print(f"  ✅ Put/Call Ratio: {ratio} (Datum: {date_str})")
                        return ratio

                # Direct score field as fallback
                score = pc_data.get("score")
                if score is not None:
                    print(f"  ✅ Put/Call Score (CNN normalized): {score}")
                    return round(float(score), 2)

        print(f"  ⚠ Put/Call CNN API Status: {resp.status_code}")
        return None
    except Exception as e:
        print(f"  ⚠ Put/Call Fehler: {e}")
        import traceback
        traceback.print_exc()
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
    # V6 point 29A: some yfinance responses (observed for e.g. 000300.SS)
    # carry NaN closes on individual rows despite `hist` otherwise being
    # non-empty -- every downstream field derived from a NaN row would
    # silently become NaN too, and Python's json.dump() happily serializes
    # a bare `NaN` token that is NOT valid JSON, breaking the ENTIRE
    # snapshot.json parse in the browser (not just this one ticker/field).
    # Same dropna-first convention already used elsewhere in this repo
    # (e.g. build_ticker_charts.compute_ticker_chart) instead of letting
    # NaN rows silently corrupt every computed metric.
    close = close.dropna()
    if len(close) < 2:
        return None
    current = close.iloc[-1]

    prev = close.iloc[-2] if len(close) >= 2 else current
    d1_pct = ((current - prev) / prev) * 100

    w1_close = close.iloc[-6] if len(close) >= 6 else close.iloc[0]
    w1_pct = ((current - w1_close) / w1_close) * 100

    high_52w = close.max()
    hi_pct = ((current - high_52w) / high_52w) * 100

    year_start = close[close.index.year == datetime.now().year]  # dropna'd `close`, not raw `hist` (point 29A)
    if len(year_start) > 0:
        ytd_start = year_start.iloc[0]
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

    # Same dropna-first guard as calc_metrics (V6 point 29A): a NaN close
    # anywhere in the window would otherwise propagate through .rolling()/
    # .ewm() into every moving average built from it.
    close = hist["Close"].dropna()
    if len(close) < 50:
        return None
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


def calc_atr_extension(hist):
    """QQQ Health price structure (V1 rebuild, point 11) — ATR14/ATR%/ATR
    Extension for SPY/QQQ. Reuses calc_true_range() imported from
    build_market_features.py so there is exactly one ATR definition
    repo-wide (True Range, simple 14-day mean, no Wilder smoothing;
    Extension = %Gain-from-SMA50 / ATR%). Returns (atr14, atr_pct,
    atr_extension), any of which may be None if there isn't enough history."""
    # Same dropna-first guard as calc_metrics (V6 point 29A): drop the whole
    # row (not just the Close cell) so High/Low/Close stay aligned, so a
    # trailing NaN close doesn't leak into `last` below and turn every
    # value here (atr_pct/atr_extension) into NaN.
    hist = hist.dropna(subset=["Close"])
    close = hist["Close"]
    if len(close) < 51:
        return None, None, None
    tr = calc_true_range(hist["High"], hist["Low"], close)
    tr_valid = tr.dropna()
    if tr_valid.shape[0] < 14:
        return None, None, None
    atr14 = float(tr_valid.iloc[-14:].mean())
    if atr14 <= 0:
        return None, None, None

    last = float(close.iloc[-1])
    sma50 = close.rolling(50).mean().iloc[-1]
    atr_pct = round(atr14 / last * 100.0, 3)
    if sma50 is None or sma50 != sma50:  # NaN check (NaN != NaN is True) without importing numpy here
        return round(atr14, 4), atr_pct, None
    sma50 = float(sma50)
    if sma50 <= 0:
        return round(atr14, 4), atr_pct, None
    gain_from_sma50_pct = (last - sma50) / sma50 * 100.0
    atr_extension = round(gain_from_sma50_pct / atr_pct, 2) if atr_pct > 0 else None
    return round(atr14, 4), atr_pct, atr_extension


def fetch_regime_data():
    """Fetch SPY and QQQ with moving averages for regime detection, plus
    ATR/ATR-Extension (feeds the QQQ Health price-structure block)."""
    print("\n🎯 Lade Regime-Daten (SPY/QQQ)...")
    regime = {}
    for symbol in ["SPY", "QQQ"]:
        print(f"  → {symbol}")
        hist = fetch_ticker_data(symbol)
        # Same NaN-close guard as calc_metrics (V6 point 29A) -- this path
        # was missed there, and a NaN QQQ price sneaking into price_structure
        # broke the browser's JSON.parse() for the ENTIRE dashboard_state.json
        # (bare `NaN` is not valid JSON), blanking Market Regime/QQQ Health/
        # Opportunities all at once even though the backend data was fine.
        if hist is not None:
            hist = hist.dropna(subset=["Close"])
        if hist is not None and len(hist) > 0:
            price = round(hist["Close"].iloc[-1], 2)
            mas = calc_moving_averages(hist)
            r = determine_regime(price, mas)
            atr14, atr_pct, atr_extension = calc_atr_extension(hist)
            regime[symbol] = {
                "price": price,
                "regime": r,
                "mas": mas,
                "atr14": atr14,
                "atr_pct": atr_pct,
                "atr_extension": atr_extension,
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Data Builder")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 YOLO Dashboard — Data Builder v3")
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

    # 3. VIX zone
    if snapshot.get("vix") and len(snapshot["vix"]) > 0:
        vix_val = snapshot["vix"][0].get("price", 0)
        snapshot["vix"][0]["zone"] = get_vix_zone(vix_val)

    # 4. QQQ McClellan + Summation + H/L
    qqq_br = fetch_qqq_breadth()
    if qqq_br:
        snapshot["qqq_breadth"] = qqq_br

    # 8. CNN Fear & Greed
    fg = fetch_fear_greed()
    if fg:
        snapshot["fear_greed"] = fg

    # 9. Put/Call Ratio (CBOE)
    pc = fetch_put_call()
    if pc:
        snapshot["put_call"] = pc

    # 10. Metadata
    snapshot["meta"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance + CNN",
    }

    # 11. Write output
    with open(out_dir / "snapshot.json", "w", encoding="utf-8") as f:
        # allow_nan=False: fail loudly here at the true source instead of
        # letting a bare NaN silently propagate into dashboard_state.json
        # and break JSON.parse() for the whole file in the browser (see the
        # dropna guard added to fetch_regime_data() for the concrete case
        # this caught: a NaN QQQ close from yfinance).
        json.dump(snapshot, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\n✅ Snapshot geschrieben → {out_dir / 'snapshot.json'}")
    print(f"   Kategorien: {len(snapshot) - 1}")
    if fg:
        print(f"   Fear & Greed: {fg['score']} ({fg['rating']})")
    if pc:
        print(f"   Put/Call: {pc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
