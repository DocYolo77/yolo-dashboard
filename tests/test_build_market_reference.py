"""
Tests for scripts/build_market_reference.py's enrich_candidates negative
caching (reliability fix, 2026-08-28): a failed per-ticker lookup (404 for
a delisted/renamed symbol, or a transient error) previously was NEVER
cached, so the SAME doomed request got retried on every single daily
run -- observed to help push a real run past the "Build full-market
features" step's 20-minute timeout. Failed lookups are now cached too, as
{"not_found": True, "cached_at": ...}, with their own shorter TTL.

Run with: pytest tests/ -v
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_market_reference as ref  # noqa: E402


def _write_cache(path, tickers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {}, "tickers": tickers}, f)


def test_failed_lookup_is_cached_as_not_found(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_reference_cache.json"
    monkeypatch.setattr(ref, "fetch_ticker_overview", lambda sym: None)

    cache = ref.enrich_candidates(["DEADCO"], cache_path)

    assert cache["DEADCO"]["not_found"] is True
    assert "cached_at" in cache["DEADCO"]
    # Downstream (build_market_features.py) reads this the same as "no
    # entry at all" via plain dict .get() -- never fabricates real fields.
    assert cache["DEADCO"].get("market_cap") is None
    assert cache["DEADCO"].get("sic_code") is None


def test_successful_lookup_is_cached_normally_alongside_a_negative_entry(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_reference_cache.json"

    def fake_fetch(sym):
        return None if sym == "DEADCO" else {"market_cap": 1_000_000_000, "sic_code": "1234"}

    monkeypatch.setattr(ref, "fetch_ticker_overview", fake_fetch)
    cache = ref.enrich_candidates(["DEADCO", "AAPL"], cache_path)

    assert cache["DEADCO"]["not_found"] is True
    assert cache["AAPL"]["market_cap"] == 1_000_000_000
    assert "not_found" not in cache["AAPL"]


def test_fresh_negative_entry_is_not_retried(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_reference_cache.json"
    recently = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    _write_cache(cache_path, {"DEADCO": {"not_found": True, "cached_at": recently}})

    calls = []
    monkeypatch.setattr(ref, "fetch_ticker_overview", lambda sym: calls.append(sym) or None)

    ref.enrich_candidates(["DEADCO"], cache_path, negative_max_age_days=3)

    assert calls == []  # 1 day old, TTL is 3 -> not re-fetched


def test_expired_negative_entry_is_retried(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_reference_cache.json"
    long_ago = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    _write_cache(cache_path, {"DEADCO": {"not_found": True, "cached_at": long_ago}})

    calls = []
    monkeypatch.setattr(ref, "fetch_ticker_overview", lambda sym: calls.append(sym) or None)

    ref.enrich_candidates(["DEADCO"], cache_path, negative_max_age_days=3)

    assert calls == ["DEADCO"]  # 5 days old, TTL is 3 -> re-fetched


def test_negative_entries_use_a_shorter_ttl_than_successful_entries_by_default():
    # The whole point of the fix: permanently-dead tickers get retried less
    # often than a real, successful enrichment result -- never MORE often.
    assert ref.NEGATIVE_ENRICH_CACHE_MAX_AGE_DAYS < ref.ENRICH_CACHE_MAX_AGE_DAYS


def test_successful_entry_still_uses_the_normal_longer_ttl_not_the_negative_one(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_reference_cache.json"
    # 4 days old: older than the negative TTL (3) but younger than the
    # normal success TTL (7, the default) -- must NOT be re-fetched.
    four_days_ago = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    _write_cache(cache_path, {"AAPL": {"market_cap": 1, "cached_at": four_days_ago}})

    calls = []
    monkeypatch.setattr(ref, "fetch_ticker_overview", lambda sym: calls.append(sym) or {"market_cap": 2})

    ref.enrich_candidates(["AAPL"], cache_path, max_age_days=7, negative_max_age_days=3)

    assert calls == []


def test_meta_reports_not_found_count(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_reference_cache.json"

    def fake_fetch(sym):
        return None if sym in ("DEAD1", "DEAD2") else {"market_cap": 1}

    monkeypatch.setattr(ref, "fetch_ticker_overview", fake_fetch)
    ref.enrich_candidates(["DEAD1", "DEAD2", "AAPL"], cache_path)

    with open(cache_path, "r", encoding="utf-8") as f:
        written = json.load(f)
    assert written["meta"]["not_found_count"] == 2
