#!/usr/bin/env python3
"""
YOLO Dashboard — Narrative Universe Reconciliation
Full-Universe Narrative Engine + Narrative Taxonomy Quality Patch (see
project briefs). Keeps the persistent semantic taxonomy
(data/taxonomy/narratives.json) in sync with the current eligible YOLO
universe (data/market_features.json).

Three modes, one script:

  --full-universe   One-time (or manually re-triggered) bulk classification
                     of every currently eligible ticker that has NEVER been
                     classified before. Batches of `classification.batch_size`
                     tickers per LLM call, growing catalog (a narrative
                     created in batch 3 is visible to batch 4+, the main
                     defense against near-duplicate narrative creation
                     across batches).

  (default)          Daily incremental reconciliation: classifies only
                     tickers that are eligible today and have NEVER been
                     classified before (true first-time entries). Re-entries
                     reuse their stored classification — zero LLM calls.
                     Exits flip `active_eligible=false` without touching the
                     classification itself (exiting the eligible universe is
                     never a semantic REMOVE).

  --cleanup          One-time Narrative Taxonomy Quality Patch: merges
                     deterministic (Level 1, normalized-name) narrative
                     duplicates, reports Level 2 near-duplicate candidates
                     for human review, strips low-confidence memberships and
                     re-runs ONLY the affected tickers through the (now
                     confidence-gated) classification pipeline. Writes
                     data/taxonomy/full_universe_cleanup_proposal.json. Never
                     a full re-classification of the whole universe.

Semantic classification NEVER sees momentum/RS/Thrust/price-performance data
— only ticker, company name, SIC code/description, and company description,
plus the existing narrative catalog.

── Quality Patch (V5): confidence gates + no forced coverage ──────────────
A Primary Narrative is only accepted if its confidence clears
membership.secondary_confidence_minimum; below that (or on an unresolvable
narrative reference) the ticker is retried up to
classification.max_retries_per_ticker times with a targeted retry prompt
that explains the specific rejection reason. If still unresolved after
retries, the ticker is left WITHOUT an active narrative and logged — this is
an accepted, correct outcome (100% NARRATIVE coverage is explicitly not
required anymore; 100% OPPORTUNITY coverage is unaffected, since
build_dashboard_states.py's Opportunity Engine iterates market_features
directly, never gated by narrative membership).

Unknown narrative-id references are NEVER silently healed into a new
narrative anymore (the old aggressive-healing fallback is removed). Instead,
resolve_narrative_reference tries a deterministic Step-A fuzzy match against
the existing catalog (token-similarity, single-plausible-match only); if
that's ambiguous or empty, the ticker goes to retry, never an auto-CREATE.

Coverage is a structural invariant, not a business requirement anymore:
validate_structural_invariants() only rejects a run that produced more than
one Primary or more than max_secondary Secondary memberships for the same
ticker (a code-bug signal), never "ticker X has no Primary" (that's now a
legitimate, retried-and-accepted outcome). A per-batch checkpoint under
.cache/ still avoids re-paying for already-successful LLM batches on retry.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm_provider  # noqa: E402

DESCRIPTION_MAX_CHARS = 320
CHECKPOINT_PATH = ".cache/narrative_classification_checkpoint.jsonl"


# ─────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, data):
    """Write via a temp file + rename so a crash mid-write never leaves a
    truncated/corrupt taxonomy on disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


def load_checkpoint(path):
    """{ticker: raw_classification_dict}, or {} if no checkpoint exists yet."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = row.get("ticker")
            if t:
                out[t] = row
    return out


def append_checkpoint(path, results):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clear_checkpoint(path):
    Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# PURE: universe / taxonomy bookkeeping
# ─────────────────────────────────────────────

def _as_number(v):
    """Tolerant-parsing rule: an LLM tool call occasionally stringifies a
    number even though the schema declares it numeric — treat non-numeric as
    missing rather than crashing the whole batch on one bad field."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_eligible_set(market_features):
    return {sym for sym, t in market_features.get("tickers", {}).items() if t.get("eligible")}


def compute_classified_tickers(taxonomy):
    """Every ticker that appears in ANY narrative's tickers dict, regardless
    of current eligibility — the taxonomy is a persistent semantic memory,
    classification survives eligibility exits."""
    return {sym for n in taxonomy.get("narratives", []) for sym in n.get("tickers", {})}


def ticker_memberships(taxonomy):
    """{ticker: [(narrative_dict, membership_dict), ...]} — a ticker can
    appear in multiple narratives (at most 1 primary + up to N secondary)."""
    out = defaultdict(list)
    for n in taxonomy.get("narratives", []):
        for sym, meta in n.get("tickers", {}).items():
            out[sym].append((n, meta))
    return out


def ticker_was_active(memberships_for_ticker):
    """A ticker's active_eligible is a fact about the TICKER, not about any
    one narrative — all of its memberships are kept in sync, so reading any
    one of them (first, arbitrarily) is representative. None (not yet
    migrated / brand new) counts as not-previously-active."""
    if not memberships_for_ticker:
        return False
    return bool(memberships_for_ticker[0][1].get("active_eligible"))


def compute_universe_changes(taxonomy, eligible_now):
    """Returns {"stayed": [...], "entered": [...], "reentered": [...],
    "exited": [...]}, each a sorted list of ticker symbols. Computed from
    each membership's CURRENTLY STORED active_eligible flag (i.e. "as of the
    previous run") before that flag gets updated for today — callers must
    call this BEFORE mark_active_eligible()."""
    classified = compute_classified_tickers(taxonomy)
    memberships = ticker_memberships(taxonomy)
    previously_active = {t for t in classified if ticker_was_active(memberships[t])}

    stayed, entered, reentered, exited = [], [], [], []
    for t in sorted(eligible_now):
        if t in previously_active:
            stayed.append(t)
        elif t in classified:
            reentered.append(t)
        else:
            entered.append(t)
    for t in sorted(previously_active - eligible_now):
        exited.append(t)

    return {"stayed": stayed, "entered": entered, "reentered": reentered, "exited": exited}


def mark_active_eligible(taxonomy, eligible_now):
    """Mechanical, LLM-free bookkeeping: every membership's active_eligible
    flag is set to whether ITS ticker is in eligible_now today. Never
    touches which narratives a ticker belongs to — exiting the eligible
    universe is never a semantic REMOVE."""
    for n in taxonomy.get("narratives", []):
        for sym, meta in n.get("tickers", {}).items():
            meta["active_eligible"] = sym in eligible_now


def migrate_membership_schema(taxonomy, today):
    """Idempotent, additive schema migration — adds assignment_priority/
    classification_source/classification_version to legacy memberships that
    predate the Full-Universe engine, without touching
    role/confidence/reason/added_at/last_reviewed_at. Entries that already
    carry assignment_priority are left completely untouched (so re-running
    this on an already-migrated taxonomy is a no-op). Returns the number of
    memberships migrated.

    Primary/secondary backfill for a legacy ticker with multiple
    memberships: the membership with role=='core' wins (ties broken by
    higher confidence, then alphabetically-first narrative id for
    determinism); everything else becomes secondary."""
    memberships = ticker_memberships(taxonomy)
    migrated = 0
    overflow = []

    for sym, entries in memberships.items():
        unmigrated = [(n, m) for n, m in entries if "assignment_priority" not in m]
        if not unmigrated:
            continue

        already_primary = [(n, m) for n, m in entries if m.get("assignment_priority") == "primary"]
        if already_primary:
            primary_pick = already_primary[0]
        else:
            ranked = sorted(
                entries,
                key=lambda nm: (0 if nm[1].get("role") == "core" else 1, -(nm[1].get("confidence") or 0), nm[0]["id"]),
            )
            primary_pick = ranked[0]

        for n, m in unmigrated:
            m["classification_source"] = m.get("classification_source", "manual_migration")
            m["classification_version"] = m.get("classification_version", 1)
            if (n, m) is primary_pick or (n["id"] == primary_pick[0]["id"] and m is primary_pick[1]):
                m["assignment_priority"] = "primary"
            else:
                m["assignment_priority"] = "secondary"
            migrated += 1

        secondary_count = sum(1 for n, m in entries if m.get("assignment_priority") == "secondary")
        if secondary_count > 2:
            overflow.append((sym, secondary_count))

    if overflow:
        print(f"  ⚠ Migration: {len(overflow)} Ticker mit >2 Secondary-Memberships nach Backfill "
              f"(manuelle Nachpruefung empfohlen): {overflow}", file=sys.stderr)
    return migrated


# ─────────────────────────────────────────────
# PURE: narrative catalog / dedup
# (Quality Patch section 7: Level 1 deterministic + Level 2 token-similarity)
# ─────────────────────────────────────────────

def _fold_simple_plural(token):
    """Conservative singular/plural fold: strips a single trailing 's' for
    tokens longer than 3 chars that don't already end in 'ss' (so
    'Wireless'/'Analytics'-style words are left alone). This is a comparison
    KEY, not a linguistic operation — it never needs to produce a real
    English word, only a stable equivalence class so 'Airline'/'Airlines'
    fold to the same key (Quality Patch point 7.1) without over-merging
    genuinely distinct concepts."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(name):
    """lowercase -> punctuation/hyphen/slash to whitespace -> per-token
    plural fold -> list of tokens (word boundaries preserved, unlike the old
    alnum-glue-everything-together normalization, which is what made
    'Healthcare Services' vs 'Healthcare Services & Facilities' impossible
    to tell apart from 'Airline' vs 'Airlines' at the token level)."""
    s = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return [_fold_simple_plural(t) for t in s.split() if t]


def normalize_narrative_name(name):
    """Deterministic Level-1 dedup key: lowercase, punctuation-insensitive,
    AND singular/plural-folded per token — 'Airline' and 'Airlines' (Quality
    Patch point 7.1) now collide, same as 'AI Infrastructure' vs
    'AI-Infrastructure' always did."""
    return "".join(_tokenize(name))


def token_similarity(name_a, name_b):
    """Level-2 near-duplicate signal: Jaccard similarity over normalized,
    plural-folded token sets. Advisory only (Quality Patch point 7.2) except
    for the single-plausible-match auto-resolve path in
    resolve_narrative_reference, which uses a much higher bar + margin than
    the (lower, purely informational) near-duplicate-candidate threshold."""
    ta, tb = set(_tokenize(name_a)), set(_tokenize(name_b))
    if not ta or not tb:
        return 0.0
    union = len(ta | tb)
    return round(len(ta & tb) / union, 4) if union else 0.0


def build_catalog_index(taxonomy):
    """{narrative_id: {"name":..., "classification_hint":...}} plus a
    {normalized_name: narrative_id} reverse index for dedup matching. Both
    are mutated in place as new narratives are created within a run, so a
    growing catalog is visible to later batches in the same run."""
    by_id = {n["id"]: {"name": n["name"], "classification_hint": n.get("classification_hint")}
              for n in taxonomy.get("narratives", [])}
    by_norm_name = {normalize_narrative_name(v["name"]): nid for nid, v in by_id.items()}
    return by_id, by_norm_name


def slugify_narrative_id(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "narrative"


def unique_narrative_id(base_slug, taken_ids):
    if base_slug not in taken_ids:
        return base_slug
    i = 2
    while f"{base_slug}_{i}" in taken_ids:
        i += 1
    return f"{base_slug}_{i}"


def _fuzzy_resolve_existing(name, catalog_by_id, threshold, margin):
    """Step-A existing-narrative resolution (Quality Patch section 5):
    scores `name` (typically an unresolved reference, underscores/hyphens
    turned to spaces) against every catalog name via token_similarity.
    Returns (resolved_id or None, status) where status is "resolved",
    "ambiguous", or "no_match". A single candidate clearing `threshold`
    resolves; multiple candidates clearing it only resolve if the top one
    leads the runner-up by at least `margin` (a genuine standout, not a coin
    flip) — otherwise ambiguous, which the caller must treat as a retry
    trigger, NEVER an auto-CREATE."""
    scored = sorted(
        ((nid, token_similarity(name, v["name"])) for nid, v in catalog_by_id.items()),
        key=lambda kv: -kv[1],
    )
    scored = [(nid, s) for nid, s in scored if s >= threshold]
    if not scored:
        return None, "no_match"
    if len(scored) == 1 or (scored[0][1] - scored[1][1]) >= margin:
        return scored[0][0], "resolved"
    return None, "ambiguous"


def find_duplicate_narrative_names(taxonomy):
    """Level-1 EXACT/NORMALIZED duplicate clusters — narratives whose
    normalize_narrative_name() collides. These are safe to auto-merge (see
    merge_duplicate_narratives) because they are, by construction, the same
    concept under case/punctuation/singular-plural folding."""
    by_norm = defaultdict(list)
    for n in taxonomy.get("narratives", []):
        by_norm[normalize_narrative_name(n["name"])].append(n["id"])
    return {norm: ids for norm, ids in by_norm.items() if len(ids) > 1}


def find_near_duplicate_candidates(taxonomy, threshold):
    """Level-2 near-duplicate candidates (Quality Patch point 14) — pairs
    with token_similarity >= threshold that are NOT already exact/normalized
    duplicates. Purely advisory: surfaced in the audit/cleanup report and the
    weekly review, NEVER auto-merged (point 7.2/18)."""
    narratives = taxonomy.get("narratives", [])
    exact_or_normalized = {
        ids[i] for ids in find_duplicate_narrative_names(taxonomy).values()
        for i in range(len(ids))
    }
    candidates = []
    for i in range(len(narratives)):
        for j in range(i + 1, len(narratives)):
            a, b = narratives[i], narratives[j]
            if a["id"] in exact_or_normalized and b["id"] in exact_or_normalized and \
               normalize_narrative_name(a["name"]) == normalize_narrative_name(b["name"]):
                continue  # already reported as an exact/normalized duplicate
            sim = token_similarity(a["name"], b["name"])
            if sim >= threshold:
                candidates.append({"narrative_a": a["id"], "narrative_b": b["id"], "similarity": sim})
    candidates.sort(key=lambda c: -c["similarity"])
    return candidates


# ─────────────────────────────────────────────
# PURE: reference resolution (Quality Patch section 5/6 — NO aggressive
# healing anymore: an unresolvable reference is a RETRY signal, never a
# silent implicit-CREATE.)
# ─────────────────────────────────────────────

DEFAULT_AUTO_RESOLVE_THRESHOLD = 0.6
DEFAULT_AUTO_RESOLVE_MARGIN = 0.08


def resolve_narrative_reference(narrative_id, is_new, new_name, new_definition, catalog_by_id, catalog_by_norm_name,
                                  auto_resolve_threshold=DEFAULT_AUTO_RESOLVE_THRESHOLD,
                                  auto_resolve_margin=DEFAULT_AUTO_RESOLVE_MARGIN):
    """Resolves one LLM-proposed narrative reference (primary or secondary)
    against the (possibly already-grown-this-run) catalog. Returns
    (resolved_id, created_new: bool, error: str|None). `error` is a stable
    machine-readable code (never a raw error object) so a caller can decide
    whether to retry — every failure mode here IS a retry trigger, there is
    no more implicit-CREATE-on-uncertainty fallback:

      "unresolvable:ambiguous"          multiple plausible existing matches
      "unresolvable:no_match"           no plausible existing match, not new
      "unresolvable:missing_new_fields" is_new=true but name/definition empty
      "unresolvable:invalid_reference"  nothing at all to work with

    Dedup happens BEFORE trusting is_new — even if the LLM claims a
    narrative is new, an EXACT/NORMALIZED match (Level 1) wins outright, and
    a single, clearly-standout NEAR match (Level 2, via _fuzzy_resolve_existing)
    is preferred over creating a near-duplicate (point 6.6/7.2 — 'existing
    bevorzugen'). Only once neither applies does this actually create a new
    narrative, and only for the is_new=true path — a reference that is NOT
    flagged is_new and doesn't resolve to anything existing is retried, it is
    never silently turned into a new narrative from the raw id (that was the
    old aggressive-healing behavior this patch explicitly removes)."""
    if narrative_id and narrative_id in catalog_by_id and not is_new:
        return narrative_id, False, None

    name_for_match = new_name if is_new else narrative_id
    if name_for_match:
        norm = normalize_narrative_name(name_for_match)
        existing = catalog_by_norm_name.get(norm)
        if existing:
            return existing, False, None

    if is_new:
        if not new_name or not new_definition:
            return None, False, "unresolvable:missing_new_fields"
        match, status = _fuzzy_resolve_existing(new_name, catalog_by_id, auto_resolve_threshold, auto_resolve_margin)
        if status == "ambiguous":
            return None, False, "unresolvable:ambiguous"
        if match:
            return match, False, None
        new_id = unique_narrative_id(slugify_narrative_id(new_name), set(catalog_by_id.keys()))
        catalog_by_id[new_id] = {"name": new_name, "classification_hint": new_definition}
        catalog_by_norm_name[normalize_narrative_name(new_name)] = new_id
        return new_id, True, None

    if not narrative_id:
        return None, False, "unresolvable:invalid_reference"

    guess_name = narrative_id.replace("_", " ").replace("-", " ").strip()
    match, status = _fuzzy_resolve_existing(guess_name, catalog_by_id, auto_resolve_threshold, auto_resolve_margin)
    if status == "ambiguous":
        return None, False, "unresolvable:ambiguous"
    if match:
        return match, False, None
    return None, False, "unresolvable:no_match"


# ─────────────────────────────────────────────
# PURE: prompt construction
# ─────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = """Du bist der semantische Klassifikations-Assistent eines Trading-Dashboards.

Deine einzige Aufgabe: jedem gegebenen Ticker genau EIN Primary Narrative und optional bis zu
zwei Secondary Narratives zuordnen — rein wirtschaftlich/geschaeftsmodellseitig.

Du bekommst NUR: Ticker, Company Name, SIC Code, SIC Description, Company Description, sowie den
bestehenden Narrative-Katalog. Du bekommst KEINE Kurs-, RS-, Thrust- oder Performance-Daten und
darfst solche Informationen auch nicht "erraten" oder in deine Begruendung einfliessen lassen.

WICHTIG:
- Primary Narrative = das wirtschaftlich wichtigste/praeziseste Narrative des Unternehmens.
- Secondary Narrative NUR bei echtem, materiellem zusaetzlichem wirtschaftlichem Exposure
  (nicht: "AI" kommt irgendwo in der Beschreibung vor).
- Bevorzuge IMMER ein bestehendes Narrative aus dem Katalog, wenn es wirtschaftlich passt.
- Erzeuge ein neues Narrative NUR wenn wirklich kein bestehendes passt. Pruefe vorher, ob es nicht
  bereits ein nahezu identisches Narrative im Katalog gibt (auch mit leicht anderem Namen, Singular/
  Plural, Bindestrich/Leerzeichen) -- nutze in dem Fall das bestehende statt ein Duplikat zu erzeugen.
- Neue Narrative brauchen einen kurzen, klaren wirtschaftlichen Definitionssatz (new_definition) UND
  einen non-empty new_name -- ohne beides ist die Klassifikation nicht verwertbar.
- Wenn du dir bei einer Zuordnung WIRKLICH nicht sicher bist (confidence < 70), gib trotzdem deine
  beste Einschaetzung mit ehrlicher confidence ab -- rate NICHT auf eine hohe Zahl hoch, nur um
  akzeptiert zu werden. Eine ehrliche niedrige confidence ist besser als eine erfundene hohe.
- Keine riesige "Other"/Sonstiges-Restkategorie -- wenn ein Unternehmen wirklich in keine bestehende
  Kategorie passt, ist ein praezises neues Narrative besser als eine Sammelkategorie.
- confidence-Felder muessen reine Zahlen 0-100 sein, niemals Text.
"""


def build_classification_context(tickers, types_ref, reference_cache, desc_max_chars=DESCRIPTION_MAX_CHARS):
    """Per-ticker fundamental/semantic context — deliberately excludes every
    momentum/RS/Thrust/price field. types_ref/reference_cache are the
    existing Massive reference caches (build_market_reference.py); zero
    additional API calls needed."""
    types_tickers = types_ref.get("tickers", {})
    ref_tickers = reference_cache.get("tickers", {})
    out = []
    for t in tickers:
        type_meta = types_tickers.get(t, {})
        ref_meta = ref_tickers.get(t, {})
        desc = (ref_meta.get("description") or "")[:desc_max_chars]
        out.append({
            "ticker": t,
            "name": type_meta.get("name"),
            "sic_code": ref_meta.get("sic_code"),
            "sic_description": ref_meta.get("sic_description"),
            "description": desc,
        })
    return out


def build_classification_user_prompt(batch_context, catalog_by_id, max_secondary):
    catalog_summary = [{"id": nid, "name": v["name"], "definition": v.get("classification_hint")}
                        for nid, v in sorted(catalog_by_id.items())]
    payload = {
        "existing_narrative_catalog": catalog_summary,
        "max_secondary_narratives": max_secondary,
        "tickers_to_classify": batch_context,
    }
    return (
        "Klassifiziere jeden Ticker unten in existing_narrative_catalog (bevorzugt) oder ein neues "
        "Narrative, falls wirklich nichts passt. max_secondary_narratives begrenzt, wie viele "
        "Secondary-Narratives du je Ticker vorschlagen darfst (0 ist erlaubt und oft richtig).\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


RETRY_REASON_EXPLANATION = {
    "low_confidence_primary": "deine vorherige confidence war zu niedrig, um akzeptiert zu werden -- "
                               "pruefe die Klassifikation nochmal gruendlich; wenn du wirklich sicherer "
                               "bist als beim letzten Mal, gib eine ehrliche hoehere confidence, sonst "
                               "eine ehrliche niedrige (das ist ein gueltiges Ergebnis)",
    "unresolvable:ambiguous": "deine vorherige primary_narrative_id war nicht eindeutig einem bestehenden "
                               "Narrative zuordenbar (mehrere aehnlich plausible Kandidaten) -- waehle "
                               "explizit EINE bestehende ID aus dem Katalog oder markiere klar is_new=true "
                               "mit eindeutigem neuem Namen",
    "unresolvable:no_match": "deine vorherige primary_narrative_id existiert nicht im Katalog und du hast "
                              "is_new nicht gesetzt -- entweder eine ECHTE bestehende ID aus dem Katalog "
                              "verwenden, oder is_new=true mit new_name und new_definition setzen",
    "unresolvable:missing_new_fields": "du hattest is_new=true gesetzt, aber new_name oder new_definition "
                                        "fehlten -- bei is_new=true MUESSEN beide gesetzt sein",
    "unresolvable:invalid_reference": "primary_narrative_id fehlte komplett -- setze entweder eine "
                                       "bestehende Katalog-ID oder is_new=true mit new_name/new_definition",
}


def build_classification_retry_user_prompt(batch_context, catalog_by_id, max_secondary, reasons_by_ticker):
    """Quality Patch section 4: a retry must NOT just repeat the identical
    prompt. Each ticker's context row carries `previous_attempt_rejected_because`
    with a specific, actionable explanation of why the previous attempt
    failed, plus an explicit instruction to re-check the catalog and prefer
    an existing narrative over inventing a new one."""
    annotated_context = [
        {**row, "previous_attempt_rejected_because":
            RETRY_REASON_EXPLANATION.get(reasons_by_ticker.get(row["ticker"]), reasons_by_ticker.get(row["ticker"]))}
        for row in batch_context
    ]
    catalog_summary = [{"id": nid, "name": v["name"], "definition": v.get("classification_hint")}
                        for nid, v in sorted(catalog_by_id.items())]
    payload = {
        "existing_narrative_catalog": catalog_summary,
        "max_secondary_narratives": max_secondary,
        "tickers_to_classify": annotated_context,
    }
    return (
        "Die vorherige Klassifikation der folgenden Ticker konnte NICHT akzeptiert werden -- siehe "
        "previous_attempt_rejected_because je Ticker fuer den konkreten Grund. Pruefe den bestehenden "
        "Narrative-Katalog erneut sorgfaeltig. Verwende ein bestehendes Narrative, wenn es wirtschaftlich "
        "vertretbar ist. Erzeuge nur dann ein neues Narrative, wenn nach dieser erneuten Pruefung "
        "wirklich kein bestehendes ausreichend passt (dann is_new=true MIT new_name UND new_definition). "
        "Eine ehrliche niedrige confidence ist ein gueltiges Ergebnis -- erfinde keine hohe Zahl.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


# ─────────────────────────────────────────────
# Batch classification (parsing is pure; the LLM call itself is not)
# ─────────────────────────────────────────────

def parse_classification_result(raw, valid_tickers):
    """Validates/cleans one raw classification dict from the LLM tool call.
    Returns (cleaned_dict, error: str|None) — never raises; a malformed
    single result must not crash the whole batch."""
    ticker = raw.get("ticker")
    if ticker not in valid_tickers:
        return None, f"unbekannter oder fehlender ticker in Klassifikationsantwort: {raw.get('ticker')!r}"
    primary_id = raw.get("primary_narrative_id")
    if not primary_id and not raw.get("primary_is_new"):
        return None, f"{ticker}: primary_narrative_id fehlt"
    primary_confidence = _as_number(raw.get("primary_confidence"))
    if primary_confidence is None:
        return None, f"{ticker}: primary_confidence ist keine gueltige Zahl"

    secondaries = []
    for s in (raw.get("secondary_narratives") or []):
        conf = _as_number(s.get("confidence"))
        nid = s.get("narrative_id")
        if not nid or conf is None:
            continue  # skip malformed secondary entries individually, don't fail the whole ticker
        secondaries.append({
            "narrative_id": nid, "is_new": bool(s.get("is_new")),
            "new_name": s.get("new_name"), "new_definition": s.get("new_definition"),
            "confidence": conf,
        })

    return {
        "ticker": ticker,
        "primary_narrative_id": primary_id,
        "primary_is_new": bool(raw.get("primary_is_new")),
        "primary_new_name": raw.get("primary_new_name"),
        "primary_new_definition": raw.get("primary_new_definition"),
        "primary_confidence": primary_confidence,
        "secondary_narratives": secondaries,
        "reasoning": raw.get("reasoning", ""),
    }, None


def apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm_name, result,
                                 membership_cfg, max_secondary, today, source,
                                 auto_resolve_threshold=DEFAULT_AUTO_RESOLVE_THRESHOLD,
                                 auto_resolve_margin=DEFAULT_AUTO_RESOLVE_MARGIN):
    """Materializes one resolved classification into the taxonomy's
    narrative->tickers dicts (creating new narrative entries as needed).

    Returns {"status": "applied"|"retry", "ticker": ..., "reason": str|None}.
    "retry" means: this ticker's Primary was rejected (confidence gate or an
    unresolvable reference) and should be retried with a fresh, targeted
    prompt — NOTHING was written for this ticker in that case (no partial
    low-confidence membership). A Secondary being rejected (confidence gate
    or unresolvable reference) never triggers "retry" for the whole ticker —
    it is silently dropped (Quality Patch point 3.2); the Primary can still
    succeed independently."""
    ticker = result["ticker"]
    min_secondary_conf = membership_cfg["secondary_confidence_minimum"]

    # Confidence gate (point 3.1): a Primary below the floor is not accepted
    # at all — better a retry (and, ultimately, "no active narrative") than
    # a forced low-confidence membership polluting the taxonomy.
    if result["primary_confidence"] < min_secondary_conf:
        return {"status": "retry", "ticker": ticker, "reason": "low_confidence_primary"}

    primary_id, created, err = resolve_narrative_reference(
        result["primary_narrative_id"], result["primary_is_new"],
        result["primary_new_name"], result["primary_new_definition"],
        catalog_by_id, catalog_by_norm_name, auto_resolve_threshold, auto_resolve_margin)
    if err:
        return {"status": "retry", "ticker": ticker, "reason": err}

    if created:
        taxonomy_by_id[primary_id] = {
            "id": primary_id, "name": catalog_by_id[primary_id]["name"], "status": "emerging",
            "classification_hint": result["primary_new_definition"],
            "created_at": today, "created_reason": f"Full-Universe classification ({source})",
            "tickers": {},
        }

    role = "core" if result["primary_confidence"] >= membership_cfg["core_confidence_minimum"] else "secondary"
    taxonomy_by_id[primary_id]["tickers"][ticker] = {
        "role": role, "assignment_priority": "primary",
        "confidence": result["primary_confidence"], "reason": result["reasoning"],
        "added_at": today, "last_reviewed_at": today, "active_eligible": True,
        "classification_source": source, "classification_version": 1,
    }

    accepted_secondary = 0
    for s in result["secondary_narratives"]:
        if accepted_secondary >= max_secondary:
            break
        if s["confidence"] < min_secondary_conf:
            continue  # point 3.2: drop silently, no retry needed just for a weak secondary
        sec_id, sec_created, sec_err = resolve_narrative_reference(
            s["narrative_id"], s["is_new"], s["new_name"], s["new_definition"],
            catalog_by_id, catalog_by_norm_name, auto_resolve_threshold, auto_resolve_margin)
        if sec_err:
            continue  # an unresolvable SECONDARY reference is dropped, not retried
        if sec_id == primary_id:
            continue  # a narrative can't be both primary and secondary for the same ticker
        if sec_created:
            taxonomy_by_id[sec_id] = {
                "id": sec_id, "name": catalog_by_id[sec_id]["name"], "status": "emerging",
                "classification_hint": s["new_definition"],
                "created_at": today, "created_reason": f"Full-Universe classification ({source})",
                "tickers": {},
            }
        sec_role = "core" if s["confidence"] >= membership_cfg["core_confidence_minimum"] else "secondary"
        taxonomy_by_id[sec_id]["tickers"][ticker] = {
            "role": sec_role, "assignment_priority": "secondary",
            "confidence": s["confidence"], "reason": result["reasoning"],
            "added_at": today, "last_reviewed_at": today, "active_eligible": True,
            "classification_source": source, "classification_version": 1,
        }
        accepted_secondary += 1

    return {"status": "applied", "ticker": ticker, "reason": None}


# ─────────────────────────────────────────────
# Structural invariant validation (NOT a coverage requirement anymore —
# see module docstring)
# ─────────────────────────────────────────────

def validate_structural_invariants(taxonomy, max_secondary):
    """Hard structural invariants that must NEVER be violated regardless of
    how many tickers ended up without a Primary (that's an accepted, logged
    outcome now, not an error): no ticker may have more than one Primary,
    and no ticker may have more than max_secondary Secondary memberships.
    A violation here indicates a code bug (e.g. a duplicate-apply), not a
    normal business outcome — it aborts the write, same as before."""
    memberships = ticker_memberships(taxonomy)
    errors = []
    for t in sorted(memberships.keys()):
        entries = memberships[t]
        primaries = [m for n, m in entries if m.get("assignment_priority") == "primary"]
        secondaries = [m for n, m in entries if m.get("assignment_priority") == "secondary"]
        if len(primaries) > 1:
            errors.append(f"{t}: {len(primaries)} Primary Narratives (muss <= 1 sein)")
        if len(secondaries) > max_secondary:
            errors.append(f"{t}: {len(secondaries)} Secondary Narratives (max {max_secondary})")
    return errors


# ─────────────────────────────────────────────
# MAIN classification loop (with targeted retries)
# ─────────────────────────────────────────────

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _call_llm_with_retries(system_prompt, user_prompt, model, label):
    last_err = None
    for attempt in range(2):
        try:
            return llm_provider.generate_ticker_classifications(system_prompt, user_prompt, model=model), None
        except llm_provider.LLMError as e:
            last_err = e
            print(f"    ⚠ {label} Versuch {attempt + 1} fehlgeschlagen: {e}", file=sys.stderr)
    return None, last_err


def classify_tickers(tickers_to_classify, taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
                      types_ref, reference_cache, cfg, checkpoint_path, source, model=None):
    """Runs the batched LLM classification loop with targeted per-ticker
    retries (Quality Patch section 4). Returns
    (applied_count, unresolved: [{"ticker":, "reason":}], hard_errors: [str])
    — `unresolved` tickers end this run with NO active narrative (an
    accepted outcome, logged for the audit report), `hard_errors` are
    LLM-call-level failures worth surfacing in the console/audit but which
    still never block the run from completing and writing.

    Uses the checkpoint to skip tickers already successfully classified in a
    prior attempt within the same run/retry — checkpointed raw results still
    go through the same parse/resolve/apply pipeline as a fresh response."""
    membership_cfg = cfg["membership"]
    cls_cfg = cfg["classification"]
    max_secondary = cls_cfg["max_secondary_narratives"]
    batch_size = cls_cfg["batch_size"]
    max_retries = cls_cfg.get("max_retries_per_ticker", 2)
    auto_resolve_threshold = cls_cfg.get("auto_resolve_similarity_threshold", DEFAULT_AUTO_RESOLVE_THRESHOLD)
    auto_resolve_margin = cls_cfg.get("auto_resolve_similarity_margin", DEFAULT_AUTO_RESOLVE_MARGIN)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    checkpoint = load_checkpoint(checkpoint_path)
    still_needed = [t for t in tickers_to_classify if t not in checkpoint]
    n_reused = len(set(checkpoint) & set(tickers_to_classify))
    if n_reused:
        print(f"  → Checkpoint gefunden: {n_reused} bereits klassifizierte Ticker werden wiederverwendet, "
              "kein erneuter LLM-Call")

    applied_count = 0
    hard_errors = []
    retry_pool = []  # [{"ticker":, "reason":}]

    def handle_one(raw, valid_tickers):
        nonlocal applied_count
        parsed, perr = parse_classification_result(raw, valid_tickers)
        if perr:
            retry_pool.append({"ticker": raw.get("ticker"), "reason": f"parse_error:{perr}"})
            return
        outcome = apply_classification_result(
            taxonomy_by_id, catalog_by_id, catalog_by_norm_name, parsed,
            membership_cfg, max_secondary, today, source, auto_resolve_threshold, auto_resolve_margin)
        if outcome["status"] == "retry":
            retry_pool.append({"ticker": outcome["ticker"], "reason": outcome["reason"]})
        else:
            applied_count += 1

    # Replay checkpointed results first (already-successful earlier batches).
    checkpointed_for_this_run = [checkpoint[t] for t in tickers_to_classify if t in checkpoint]
    for raw in checkpointed_for_this_run:
        handle_one(raw, set(tickers_to_classify))

    batches = list(chunked(still_needed, batch_size))
    n_batches = len(batches)
    for i, batch in enumerate(batches, 1):
        print(f"  → Batch {i}/{n_batches} ({len(batch)} Ticker)...")
        context = build_classification_context(batch, types_ref, reference_cache)
        user_prompt = build_classification_user_prompt(context, catalog_by_id, max_secondary)

        raw_results, err = _call_llm_with_retries(CLASSIFICATION_SYSTEM_PROMPT, user_prompt, model, f"Batch {i}")
        if raw_results is None:
            hard_errors.append(f"Batch {i} ({batch[0]}..{batch[-1]}): LLM-Aufruf endgueltig fehlgeschlagen: {err}")
            for t in batch:
                retry_pool.append({"ticker": t, "reason": "llm_call_failed"})
            continue

        append_checkpoint(checkpoint_path, raw_results)
        for raw in raw_results:
            handle_one(raw, set(batch))

    # Targeted retry rounds (section 4): a distinct, reason-annotated prompt
    # per round, never a bare repeat of the original prompt.
    for round_num in range(1, max_retries + 1):
        if not retry_pool:
            break
        print(f"  → Retry-Runde {round_num}/{max_retries}: {len(retry_pool)} Ticker")
        reasons_by_ticker = {x["ticker"]: x["reason"] for x in retry_pool if x["ticker"]}
        retry_tickers = sorted(reasons_by_ticker.keys())
        retry_pool = [x for x in retry_pool if not x["ticker"]]  # keep unresolvable-ticker=None entries out of the loop

        for batch in chunked(retry_tickers, batch_size):
            context = build_classification_context(batch, types_ref, reference_cache)
            user_prompt = build_classification_retry_user_prompt(context, catalog_by_id, max_secondary, reasons_by_ticker)
            raw_results, err = _call_llm_with_retries(
                CLASSIFICATION_SYSTEM_PROMPT, user_prompt, model, f"Retry-Runde {round_num}")
            if raw_results is None:
                hard_errors.append(f"Retry-Runde {round_num} Batch ({batch[0]}..{batch[-1]}): "
                                    f"LLM-Aufruf endgueltig fehlgeschlagen: {err}")
                for t in batch:
                    retry_pool.append({"ticker": t, "reason": reasons_by_ticker[t]})
                continue
            append_checkpoint(checkpoint_path, raw_results)
            for raw in raw_results:
                handle_one(raw, set(batch))

    unresolved = [x for x in retry_pool if x["ticker"]]
    if unresolved:
        print(f"  → {len(unresolved)} Ticker nach {max_retries} Retry-Runden weiterhin ohne aktives "
              "Narrative (akzeptiertes Ergebnis, kein Fehler)")

    return applied_count, unresolved, hard_errors


# ─────────────────────────────────────────────
# Audit / sanity report
# ─────────────────────────────────────────────

def build_audit_state(as_of_date, eligible_now, classified_eligible_count, changes,
                       narratives_before, narratives_after, unresolved, hard_errors):
    coverage_pct = round(classified_eligible_count / len(eligible_now) * 100, 1) if eligible_now else 100.0
    return {
        "as_of_date": as_of_date,
        "eligible_count": len(eligible_now),
        "classified_eligible_count": classified_eligible_count,
        "coverage_pct": coverage_pct,
        "entered": changes["entered"],
        "reentered": changes["reentered"],
        "exited": changes["exited"],
        "newly_classified": sorted(set(changes["entered"])),
        "new_narratives_created": sorted(set(narratives_after) - set(narratives_before)),
        "unresolved_after_retries": unresolved,
        "classification_errors": hard_errors,
    }


def print_sanity_report(eligible_now, changes, taxonomy_before_ids, taxonomy_after,
                         unresolved, hard_errors, active_stats, sample_tickers,
                         near_duplicate_candidates=None):
    memberships = ticker_memberships(taxonomy_after)
    classified_now = compute_classified_tickers(taxonomy_after)

    print("\n" + "=" * 60)
    print("🔍 FULL-UNIVERSE NARRATIVE ENGINE — SANITY REPORT")
    print("=" * 60)

    print(f"\nELIGIBLE UNIVERSE: {len(eligible_now)}")

    classified_eligible = eligible_now & classified_now
    coverage_pct = round(len(classified_eligible) / len(eligible_now) * 100, 1) if eligible_now else 100.0
    primary_n = sum(1 for t in eligible_now if any(m.get("assignment_priority") == "primary" for _, m in memberships[t]))
    secondary_n = sum(len([m for _, m in memberships[t] if m.get("assignment_priority") == "secondary"]) for t in eligible_now)
    print(f"CLASSIFICATION: classified={len(classified_eligible)} | coverage={coverage_pct}% | "
          f"primary_assignments={primary_n} | secondary_assignments={secondary_n}")
    print(f"  (100% Narrative Coverage ist NICHT mehr Pflicht -- {len(eligible_now) - primary_n} eligible "
          "Ticker ohne Primary sind ein gueltiges Ergebnis, kein Fehler)")

    print(f"UNIVERSE CHANGES: stayed={len(changes['stayed'])} | entered={len(changes['entered'])} | "
          f"reentered={len(changes['reentered'])} | exited={len(changes['exited'])}")

    n_ids_after = {n["id"] for n in taxonomy_after.get("narratives", [])}
    created = sorted(n_ids_after - taxonomy_before_ids)
    print(f"NARRATIVES: before={len(taxonomy_before_ids)} | after={len(n_ids_after)} | neu erzeugt={len(created)}")
    if created:
        print(f"  neue Narratives: {created}")

    if active_stats:
        print(f"\nACTIVE NARRATIVE COVERAGE (min. {active_stats['minimum_active_narrative_members']} eligible Mitglieder):")
        print(f"  eligible_with_active_narrative={active_stats['eligible_with_active_narrative']} | "
              f"eligible_without_active_narrative={active_stats['eligible_without_active_narrative']} | "
              f"active_narrative_coverage_pct={active_stats['active_narrative_coverage_pct']}%")
        print(f"  active_narrative_count={active_stats['active_narrative_count']} | "
              f"undersized_narrative_count={active_stats['undersized_narrative_count']}")

    dupes = find_duplicate_narrative_names(taxonomy_after)
    empty_narratives = [n["id"] for n in taxonomy_after.get("narratives", []) if not n["tickers"]]
    print(f"\nQUALITY CHECK: leere Narratives={len(empty_narratives)} | "
          f"Duplicate-Name-Cluster={len(dupes)} | Unresolved-nach-Retries={len(unresolved)} | "
          f"LLM-Fehler={len(hard_errors)}")
    if dupes:
        print(f"  Duplicate-Name-Cluster (sollten via --cleanup gemergt sein): {dupes}")
    if near_duplicate_candidates:
        print(f"  Near-Duplicate Candidates (nur Review, {len(near_duplicate_candidates)}): "
              f"{near_duplicate_candidates[:10]}")

    print("\nSTICHPROBEN:")
    for t in sample_tickers:
        entries = memberships.get(t)
        if not entries:
            print(f"  {t}: NICHT KLASSIFIZIERT")
            continue
        primary = next((n["name"] for n, m in entries if m.get("assignment_priority") == "primary"), None)
        secondary = [n["name"] for n, m in entries if m.get("assignment_priority") == "secondary"]
        reason = next((m.get("reason") for n, m in entries if m.get("assignment_priority") == "primary"), "")
        print(f"  {t}: Primary={primary or '—'} | Secondary={secondary or '-'}")
        if reason:
            print(f"       Begruendung: {reason[:180]}")
    print("=" * 60)


# ─────────────────────────────────────────────
# CLEANUP MODE (Quality Patch sections 14-18): one-time deterministic
# Level-1 duplicate merge + targeted reclassification of low-confidence /
# duplicate-narrative / id-derived-name memberships. NEVER a full rebuild.
# ─────────────────────────────────────────────

def _looks_id_derived(narrative):
    """Heuristic flag for narratives that are artifacts of the OLD
    (now-removed) aggressive-healing fallback: their name is exactly the
    mechanical title-case transform of their own id, AND they were created
    by the Full-Universe classifier. Imperfect by nature (a legitimate
    LLM-authored name can coincidentally match this pattern) — used only to
    flag REVIEW/reclassification candidates, never to silently delete
    anything."""
    if "Full-Universe classification" not in (narrative.get("created_reason") or ""):
        return False
    mechanical = narrative["id"].replace("_", " ").replace("-", " ").strip().title()
    return narrative.get("name") == mechanical


def merge_duplicate_narratives(taxonomy):
    """Deterministically merges every Level-1 (exact/normalized-name)
    duplicate cluster into a single surviving narrative, transferring every
    membership (Quality Patch point 18). Survivor selection: most current
    eligible-agnostic membership count (i.e. most tickers overall — a proxy
    for "the more established entry" without requiring market_features
    here), tie-broken by earliest created_at, then alphabetical id — fully
    deterministic, no LLM involved. Per-ticker conflicts (same ticker in
    both narratives) are resolved by: Primary beats Secondary, then higher
    confidence, then role core beats secondary, then earlier added_at.
    Returns (merged_taxonomy, merge_log: [{"into":, "merged_from": [...]}])."""
    clusters = find_duplicate_narrative_names(taxonomy)
    if not clusters:
        return taxonomy, []

    by_id = {n["id"]: n for n in taxonomy["narratives"]}
    merge_log = []
    removed_ids = set()

    def survivor_key(nid):
        n = by_id[nid]
        return (-len(n.get("tickers", {})), n.get("created_at") or "9999-99-99", nid)

    def membership_priority(m):
        # Lower sorts "better" (wins ties) — Primary(0) < Secondary(1);
        # higher confidence wins; core(0) < secondary(1); earlier added_at wins.
        return (
            0 if m.get("assignment_priority") == "primary" else 1,
            -(m.get("confidence") or 0),
            0 if m.get("role") == "core" else 1,
            m.get("added_at") or "9999-99-99",
        )

    for norm, ids in clusters.items():
        ordered = sorted(ids, key=survivor_key)
        survivor_id = ordered[0]
        losers = ordered[1:]
        survivor = by_id[survivor_id]

        for loser_id in losers:
            loser = by_id[loser_id]
            for sym, m in loser.get("tickers", {}).items():
                existing = survivor["tickers"].get(sym)
                if existing is None or membership_priority(m) < membership_priority(existing):
                    survivor["tickers"][sym] = m
            removed_ids.add(loser_id)

        merge_log.append({"into": survivor_id, "merged_from": losers,
                           "normalized_name": norm, "name": survivor["name"]})

    taxonomy["narratives"] = [n for n in taxonomy["narratives"] if n["id"] not in removed_ids]
    return taxonomy, merge_log


def find_low_confidence_memberships(taxonomy, membership_cfg):
    """Buckets every CURRENT membership by confidence vs. the same
    thresholds classification/daily reconciliation now enforces going
    forward (Quality Patch point 16). Returns
    {"primary_below_minimum": [...], "secondary_below_minimum": [...],
     "primary_70_to_84": [...], "primary_85_plus": [...]}, each a list of
    {"narrative_id":, "ticker":, "confidence":}."""
    core_min = membership_cfg["core_confidence_minimum"]
    sec_min = membership_cfg["secondary_confidence_minimum"]
    buckets = {"primary_below_minimum": [], "secondary_below_minimum": [],
               "primary_70_to_84": [], "primary_85_plus": []}
    for n in taxonomy.get("narratives", []):
        for sym, m in n.get("tickers", {}).items():
            conf = m.get("confidence") or 0
            priority = m.get("assignment_priority")
            row = {"narrative_id": n["id"], "ticker": sym, "confidence": conf}
            if priority == "primary":
                if conf < sec_min:
                    buckets["primary_below_minimum"].append(row)
                elif conf < core_min:
                    buckets["primary_70_to_84"].append(row)
                else:
                    buckets["primary_85_plus"].append(row)
            elif priority == "secondary" and conf < sec_min:
                buckets["secondary_below_minimum"].append(row)
    return buckets


def remove_membership(taxonomy, narrative_id, ticker):
    for n in taxonomy.get("narratives", []):
        if n["id"] == narrative_id:
            n.get("tickers", {}).pop(ticker, None)
            return


def compute_active_undersized_stats(taxonomy, eligible_now, minimum_active_members):
    """Same active/undersized rule build_narratives.py applies for the
    dashboard (Quality Patch section 8/19), computed here purely for the
    reconciliation/cleanup audit report — NOT the source of truth for the
    dashboard itself (that stays build_narratives.py, computed fresh every
    run from market_features). eligible member counts here are PRIMARY-OR-
    SECONDARY membership (any current membership), matching build_narratives.py's
    `members` filter."""
    active_ids, undersized = set(), []
    primary_narrative_of = {}
    for n in taxonomy.get("narratives", []):
        n_eligible = sum(1 for sym in n.get("tickers", {}) if sym in eligible_now)
        if n_eligible >= minimum_active_members:
            active_ids.add(n["id"])
        else:
            undersized.append({"id": n["id"], "name": n["name"], "eligible_member_count": n_eligible})
        for sym, m in n.get("tickers", {}).items():
            if m.get("assignment_priority") == "primary":
                primary_narrative_of[sym] = n["id"]

    with_active = sum(1 for t in eligible_now if primary_narrative_of.get(t) in active_ids)
    without_active = len(eligible_now) - with_active
    pct = round(with_active / len(eligible_now) * 100, 1) if eligible_now else 100.0
    return {
        "minimum_active_narrative_members": minimum_active_members,
        "eligible_with_active_narrative": with_active,
        "eligible_without_active_narrative": without_active,
        "active_narrative_coverage_pct": pct,
        "active_narrative_count": len(active_ids),
        "undersized_narrative_count": len(undersized),
        "undersized_narratives": sorted(undersized, key=lambda u: -u["eligible_member_count"]),
    }


def run_cleanup(taxonomy, market_features, types_ref, reference_cache, cfg, checkpoint_path, today, model=None):
    """Orchestrates the one-time Narrative Taxonomy Quality Patch cleanup
    (sections 14-18): merge Level-1 duplicates, identify (never auto-merge)
    Level-2 near-duplicates, strip low-confidence memberships, and
    reclassify ONLY the affected tickers through the (now confidence-gated,
    non-healing) pipeline. Returns (taxonomy, proposal_dict)."""
    membership_cfg = cfg["membership"]
    near_dup_threshold = cfg["classification"].get("near_duplicate_similarity_threshold", 0.5)
    min_active = cfg["classification"].get("minimum_active_narrative_members", 5)

    narratives_before_ids = {n["id"] for n in taxonomy["narratives"]}
    exact_before = find_duplicate_narrative_names(taxonomy)
    near_dup_before = find_near_duplicate_candidates(taxonomy, near_dup_threshold)

    taxonomy, merge_log = merge_duplicate_narratives(taxonomy)
    print(f"  → Level-1 Duplicate gemergt: {len(merge_log)} Cluster")

    id_derived = [n["id"] for n in taxonomy["narratives"] if _looks_id_derived(n)]
    if id_derived:
        print(f"  → {len(id_derived)} vermutlich id-abgeleitete Narrative-Namen (Alt-Healing-Artefakte) "
              f"zur Reklassifikation markiert: {id_derived}")

    buckets = find_low_confidence_memberships(taxonomy, membership_cfg)
    for row in buckets["secondary_below_minimum"]:
        remove_membership(taxonomy, row["narrative_id"], row["ticker"])
    print(f"  → {len(buckets['secondary_below_minimum'])} Secondary-Memberships unter Mindest-Confidence entfernt")

    tickers_to_reclassify = sorted({row["ticker"] for row in buckets["primary_below_minimum"]}
                                    | {sym for n in taxonomy["narratives"] if n["id"] in id_derived
                                       for sym in n.get("tickers", {})})
    for row in buckets["primary_below_minimum"]:
        remove_membership(taxonomy, row["narrative_id"], row["ticker"])
    for nid in id_derived:
        n = next(n for n in taxonomy["narratives"] if n["id"] == nid)
        for sym in list(n.get("tickers", {}).keys()):
            n["tickers"].pop(sym, None)
    print(f"  → {len(tickers_to_reclassify)} Ticker zur gezielten Reklassifikation markiert "
          "(Low-Confidence-Primary und/oder Alt-Healing-Artefakt)")

    catalog_by_id, catalog_by_norm_name = build_catalog_index(taxonomy)
    taxonomy_by_id = {n["id"]: n for n in taxonomy["narratives"]}
    applied_count, unresolved, hard_errors = (0, [], [])
    if tickers_to_reclassify:
        applied_count, unresolved, hard_errors = classify_tickers(
            tickers_to_reclassify, taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
            types_ref, reference_cache, cfg, checkpoint_path, "cleanup_reclassification", model)
        taxonomy["narratives"] = list(taxonomy_by_id.values())
    print(f"  → Reklassifikation: {applied_count} erfolgreich, {len(unresolved)} weiterhin ohne "
          "aktives Narrative (akzeptiert)")

    # A narrative left with literally ZERO members (not just undersized —
    # genuinely empty) after stripping low-confidence/id-derived memberships
    # AND running reclassification is dead weight with no semantic content
    # left; prune it. This is safe because reclassification already had the
    # full opportunity to route any of its former tickers back into it (the
    # narrative was still present in the catalog throughout classify_tickers
    # above) — an empty result here means nothing, past or present, actually
    # belongs to it anymore.
    pruned_empty = [n["id"] for n in taxonomy["narratives"] if not n.get("tickers")]
    if pruned_empty:
        taxonomy["narratives"] = [n for n in taxonomy["narratives"] if n.get("tickers")]
        print(f"  → {len(pruned_empty)} vollstaendig leere Narrative nach Cleanup entfernt: {pruned_empty}")

    eligible_now = compute_eligible_set(market_features)
    active_stats = compute_active_undersized_stats(taxonomy, eligible_now, min_active)
    near_dup_after = find_near_duplicate_candidates(taxonomy, near_dup_threshold)

    proposal = {
        "created_at": today,
        "exact_duplicate_names_before": exact_before,
        "near_duplicate_candidates_before": near_dup_before,
        "merged": merge_log,
        "near_duplicate_candidates_remaining": near_dup_after,
        "id_derived_name_candidates": id_derived,
        "low_confidence_memberships": {
            "primary_below_minimum": buckets["primary_below_minimum"],
            "secondary_below_minimum_removed": buckets["secondary_below_minimum"],
            "primary_70_to_84": buckets["primary_70_to_84"],
            "primary_85_plus_count": len(buckets["primary_85_plus"]),
        },
        "reclassified_tickers": tickers_to_reclassify,
        "reclassified_applied_count": applied_count,
        "unresolved_after_retries": unresolved,
        "classification_errors": hard_errors,
        "pruned_empty_narratives": pruned_empty,
        "undersized_narratives": active_stats["undersized_narratives"],
        "active_narrative_count": active_stats["active_narrative_count"],
        "undersized_narrative_count": active_stats["undersized_narrative_count"],
    }
    return taxonomy, proposal, active_stats, narratives_before_ids


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Narrative Universe Reconciliation")
    parser.add_argument("--market-features", default="data/market_features.json")
    parser.add_argument("--taxonomy", default="data/taxonomy/narratives.json")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--types-ref", default="data/taxonomy/market_reference_types.json")
    parser.add_argument("--reference-cache", default="data/taxonomy/market_reference_cache.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--full-universe", action="store_true",
                         help="Classify EVERY currently-unclassified eligible ticker, not just today's new entries")
    parser.add_argument("--cleanup", action="store_true",
                         help="One-time Narrative Taxonomy Quality Patch cleanup (merge duplicates, "
                              "reclassify low-confidence/id-derived memberships) instead of a normal run")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config.classification.batch_size")
    parser.add_argument("--dry-run", action="store_true", help="Compute + report, never write the taxonomy")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mode_label = "[CLEANUP]" if args.cleanup else ("[FULL-UNIVERSE]" if args.full_universe else "[daily incremental]")
    print("=" * 60)
    print(f"🧭 YOLO Dashboard — Narrative Universe Reconciliation ({today}) {mode_label}")
    print("=" * 60)

    market_features = load_json(args.market_features)
    taxonomy = load_json(args.taxonomy)
    cfg = load_json(args.config)
    types_ref = load_json(args.types_ref)
    reference_cache = load_json(args.reference_cache)

    if args.batch_size:
        cfg["classification"]["batch_size"] = args.batch_size

    migrated = migrate_membership_schema(taxonomy, today)
    if migrated:
        print(f"  → Schema-Migration: {migrated} Legacy-Memberships mit assignment_priority nachgeruestet")

    min_active = cfg["classification"].get("minimum_active_narrative_members", 5)

    if args.cleanup:
        taxonomy, proposal, active_stats, narratives_before_ids = run_cleanup(
            taxonomy, market_features, types_ref, reference_cache, cfg, args.checkpoint, today)

        structural_errors = validate_structural_invariants(taxonomy, cfg["classification"]["max_secondary_narratives"])
        if structural_errors:
            print(f"\n❌ Struktur-Invarianten verletzt ({len(structural_errors)}) — Taxonomie wird NICHT geschrieben:",
                  file=sys.stderr)
            for e in structural_errors[:50]:
                print(f"   - {e}", file=sys.stderr)
            sys.exit(1)

        eligible_now = compute_eligible_set(market_features)
        sample_tickers = [t for t in ["NVDA", "DELL", "MU", "SNDK", "CRWD", "FTNT"] if t in eligible_now]
        sample_tickers += sorted((eligible_now - set(sample_tickers)))[:3]
        print_sanity_report(eligible_now, {"stayed": [], "entered": [], "reentered": [], "exited": []},
                             narratives_before_ids, taxonomy, proposal["unresolved_after_retries"],
                             proposal["classification_errors"], active_stats, sample_tickers,
                             proposal["near_duplicate_candidates_remaining"])

        if args.dry_run:
            print("\n(dry-run: Taxonomie/Proposal nicht geschrieben)")
            return

        save_json_atomic(args.taxonomy, taxonomy)
        print(f"\n✅ Taxonomie geschrieben → {args.taxonomy}")
        proposal_path = Path(args.out_dir) / "taxonomy" / "full_universe_cleanup_proposal.json"
        save_json_atomic(proposal_path, proposal)
        print(f"✅ Cleanup-Proposal geschrieben → {proposal_path}")
        clear_checkpoint(args.checkpoint)
        print("=" * 60)
        return

    eligible_now = compute_eligible_set(market_features)
    changes = compute_universe_changes(taxonomy, eligible_now)
    print(f"  → Eligible: {len(eligible_now)} | Stayed: {len(changes['stayed'])} | "
          f"Entered: {len(changes['entered'])} | Reentered: {len(changes['reentered'])} | Exited: {len(changes['exited'])}")

    classified = compute_classified_tickers(taxonomy)
    if args.full_universe:
        tickers_to_classify = sorted(eligible_now - classified)
        print(f"  → Full-Universe-Modus: {len(tickers_to_classify)} noch unklassifizierte eligible Ticker")
    else:
        tickers_to_classify = sorted(t for t in changes["entered"] if t not in classified)
        print(f"  → Inkrementeller Modus: {len(tickers_to_classify)} echte Neuzugaenge zu klassifizieren")

    narratives_before_ids = {n["id"] for n in taxonomy["narratives"]}
    catalog_by_id, catalog_by_norm_name = build_catalog_index(taxonomy)
    taxonomy_by_id = {n["id"]: n for n in taxonomy["narratives"]}

    unresolved, hard_errors = [], []
    if tickers_to_classify:
        source = "full_universe_run" if args.full_universe else "daily_reconciliation"
        applied_count, unresolved, hard_errors = classify_tickers(
            tickers_to_classify, taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
            types_ref, reference_cache, cfg, args.checkpoint, source)
        print(f"  → {applied_count}/{len(tickers_to_classify)} Ticker klassifiziert, "
              f"{len(unresolved)} unresolved, {len(hard_errors)} LLM-Fehler")
        taxonomy["narratives"] = list(taxonomy_by_id.values())

    mark_active_eligible(taxonomy, eligible_now)

    structural_errors = validate_structural_invariants(taxonomy, cfg["classification"]["max_secondary_narratives"])

    sample_tickers = [t for t in ["NVDA", "DELL", "MU", "SNDK", "CRWD", "FTNT"] if t in eligible_now]
    sample_tickers += sorted((eligible_now - set(sample_tickers)))[:3]
    active_stats = compute_active_undersized_stats(taxonomy, eligible_now, min_active)

    if structural_errors:
        print(f"\n❌ Struktur-Invarianten verletzt ({len(structural_errors)}) — Taxonomie wird NICHT geschrieben:",
              file=sys.stderr)
        for e in structural_errors[:50]:
            print(f"   - {e}", file=sys.stderr)
        print_sanity_report(eligible_now, changes, narratives_before_ids, taxonomy, unresolved, hard_errors,
                             active_stats, sample_tickers)
        sys.exit(1)

    print_sanity_report(eligible_now, changes, narratives_before_ids, taxonomy, unresolved, hard_errors,
                         active_stats, sample_tickers)

    if args.dry_run:
        print("\n(dry-run: Taxonomie nicht geschrieben)")
        return

    save_json_atomic(args.taxonomy, taxonomy)
    print(f"\n✅ Taxonomie geschrieben → {args.taxonomy}")

    audit = build_audit_state(today, eligible_now, len(eligible_now & compute_classified_tickers(taxonomy)),
                               changes, narratives_before_ids, {n["id"] for n in taxonomy["narratives"]},
                               unresolved, hard_errors)
    audit.update({
        "eligible_with_active_narrative": active_stats["eligible_with_active_narrative"],
        "eligible_without_active_narrative": active_stats["eligible_without_active_narrative"],
        "active_narrative_coverage_pct": active_stats["active_narrative_coverage_pct"],
        "active_narrative_count": active_stats["active_narrative_count"],
        "undersized_narrative_count": active_stats["undersized_narrative_count"],
    })
    audit_path = Path(args.out_dir) / "narrative_universe_state.json"
    save_json_atomic(audit_path, audit)
    print(f"✅ Audit-Report geschrieben → {audit_path}")

    clear_checkpoint(args.checkpoint)
    print("=" * 60)


if __name__ == "__main__":
    main()
