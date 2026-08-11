"""
Tests for scripts/build_narratives.load_taxonomy (structured taxonomy +
legacy fallback) and scripts/migrate_taxonomy (Multi-Membership, migrated
data structure). No network required.
Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import load_taxonomy  # noqa: E402
from migrate_taxonomy import migrate  # noqa: E402


def test_load_taxonomy_structured_format_multi_membership(tmp_path):
    taxonomy_path = tmp_path / "narratives.json"
    taxonomy_path.write_text(json.dumps({
        "narratives": [
            {"id": "semis", "name": "Semiconductors", "status": "active",
             "tickers": {"NVDA": {"role": "core"}, "AMD": {"role": "core"}}},
            {"id": "ai_infra", "name": "AI Infrastructure", "status": "emerging",
             "tickers": {"NVDA": {"role": "core"}, "VRT": {"role": "secondary"}}},
        ]
    }), encoding="utf-8")

    narratives, universe = load_taxonomy(str(taxonomy_path))
    assert universe == ["AMD", "NVDA", "VRT"]
    # NVDA belongs to two narratives at once -> Multi-Membership.
    nvda_memberships = [n["id"] for n in narratives if "NVDA" in n["tickers"]]
    assert set(nvda_memberships) == {"semis", "ai_infra"}
    statuses = {n["id"]: n["status"] for n in narratives}
    assert statuses == {"semis": "active", "ai_infra": "emerging"}


def test_load_taxonomy_falls_back_to_legacy_flat_list(tmp_path):
    legacy_path = tmp_path / "narratives_map.json"
    legacy_path.write_text(json.dumps({
        "narratives": [{"id": "legacy_basket", "name": "Legacy", "tickers": ["AAA", "BBB"]}]
    }), encoding="utf-8")
    missing_path = tmp_path / "does_not_exist.json"

    narratives, universe = load_taxonomy(str(missing_path), legacy_path=str(legacy_path))
    assert universe == ["AAA", "BBB"]
    assert narratives[0]["status"] == "active"  # default when legacy format has no status


def test_migrate_preserves_multi_membership_and_sets_defaults():
    legacy = {"narratives": [
        {"id": "a", "name": "A", "tickers": ["X", "Y"]},
        {"id": "b", "name": "B", "tickers": ["Y", "Z"]},
    ]}
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(legacy, f)
        legacy_path = f.name

    migrated = migrate(Path(legacy_path), "2026-08-11")
    by_id = {n["id"]: n for n in migrated["narratives"]}
    assert set(by_id["a"]["tickers"].keys()) == {"X", "Y"}
    assert set(by_id["b"]["tickers"].keys()) == {"Y", "Z"}
    # Y is a member of both -> Multi-Membership survives migration.
    assert "Y" in by_id["a"]["tickers"] and "Y" in by_id["b"]["tickers"]
    for entry in list(by_id["a"]["tickers"].values()) + list(by_id["b"]["tickers"].values()):
        assert entry["role"] == "core"
        assert entry["confidence"] == 85
        assert entry["added_at"] == "2026-08-11"
        assert entry["last_reviewed_at"] == "2026-08-11"
    assert migrated["schema_version"] == 1
