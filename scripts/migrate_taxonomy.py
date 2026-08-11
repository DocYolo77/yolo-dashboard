#!/usr/bin/env python3
"""
YOLO Dashboard — One-time Taxonomy Migration
Migrates the legacy manually-curated data/narratives_map.json (id, name,
tickers[]) into the structured Source of Truth data/taxonomy/narratives.json
(id, name, status, tickers: {SYMBOL: {role, confidence, reason, added_at,
last_reviewed_at}}) and writes the first dated history snapshot.

Run once: python scripts/migrate_taxonomy.py
Safe to re-run — it only overwrites data/taxonomy/narratives.json if
--force is passed, so an already-migrated (and since hand-edited or
weekly-review-updated) taxonomy is never silently clobbered.

data/narratives_map.json is intentionally NOT deleted: it remains as the
seed record of the original manually-curated taxonomy.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROLE = "core"
DEFAULT_CONFIDENCE = 85
DEFAULT_REASON = "Migrated from legacy narratives_map.json (manual curation, pre-engine)"


def migrate(legacy_path: Path, migration_date: str):
    with open(legacy_path, "r", encoding="utf-8") as f:
        legacy = json.load(f)

    narratives = []
    for n in legacy["narratives"]:
        tickers = {}
        for sym in n["tickers"]:
            tickers[sym] = {
                "role": DEFAULT_ROLE,
                "confidence": DEFAULT_CONFIDENCE,
                "reason": DEFAULT_REASON,
                "added_at": migration_date,
                "last_reviewed_at": migration_date,
            }
        narratives.append({
            "id": n["id"],
            "name": n["name"],
            "status": "active",
            "classification_hint": None,
            "tickers": tickers,
        })

    return {
        "_comment": "Source of Truth fuer Narrative-Klassifikation und Memberships. "
                    "Enthaelt KEINE taeglich wechselnden Strength-/Thrust-Werte (die "
                    "liegen in data/narratives.json). Wird durch akzeptierte Weekly-"
                    "Review-PRs veraendert (data/taxonomy/proposals/*.json), nie direkt "
                    "vom LLM ueberschrieben.",
        "schema_version": 1,
        "migrated_from": str(legacy_path),
        "migrated_at": migration_date,
        "narratives": narratives,
    }


def write_history_snapshot(taxonomy: dict, history_dir: Path, date_str: str):
    history_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = history_dir / f"{date_str}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    return snapshot_path


def main():
    parser = argparse.ArgumentParser(description="Migrate narratives_map.json -> data/taxonomy/narratives.json")
    parser.add_argument("--legacy", default="data/narratives_map.json")
    parser.add_argument("--out", default="data/taxonomy/narratives.json")
    parser.add_argument("--history-dir", default="data/taxonomy/history")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing taxonomy file")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"FATAL: {out_path} existiert bereits. Nutze --force zum Ueberschreiben.", file=sys.stderr)
        sys.exit(1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    taxonomy = migrate(Path(args.legacy), today)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    print(f"✅ Taxonomie geschrieben → {out_path} ({len(taxonomy['narratives'])} Narrative)")

    snap = write_history_snapshot(taxonomy, Path(args.history_dir), today)
    print(f"✅ History-Snapshot geschrieben → {snap}")


if __name__ == "__main__":
    main()
