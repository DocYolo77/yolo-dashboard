#!/usr/bin/env python3
"""
YOLO Dashboard — LLM Provider Abstraction
Isolates the one place in the pipeline that talks to a specific LLM vendor,
so a future model/provider swap does not require touching
scripts/review_narratives.py. Implemented with plain `requests` against the
Anthropic Messages API (no `anthropic` SDK dependency, consistent with the
rest of this repo's requests-only style).

The API key is read exclusively from the ANTHROPIC_API_KEY environment
variable (populated from the GitHub Actions secret of the same name) — it
is never read from a file, never logged, and never written to any output.

Structured output is enforced via Anthropic tool-use with a forced
tool_choice: the model MUST call `propose_narrative_changes` with arguments
matching CHANGE_SET_TOOL_SCHEMA below, so a free-form "here is a whole new
taxonomy" text response is not a valid response shape in the first place.
"""

import json
import os
import sys

import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = os.environ.get("NARRATIVE_REVIEW_MODEL", "claude-sonnet-5")

# Mirrors the CREATE/ADD/REMOVE/STATUS_CHANGE/MERGE_PROPOSAL/SPLIT_PROPOSAL
# change shape documented in validate_narrative_proposal.py. Kept
# intentionally permissive on types (validator does the strict checking) —
# the tool schema's job is to force "a list of structured change objects",
# not to fully re-implement the business rules.
CHANGE_SET_TOOL_SCHEMA = {
    "name": "propose_narrative_changes",
    "description": "Submit a structured change set against the current narrative taxonomy. "
                    "Never submit a full replacement taxonomy — only the deltas.",
    "input_schema": {
        "type": "object",
        "required": ["changes"],
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["action", "semantic_evidence", "reason"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["CREATE", "ADD", "REMOVE", "STATUS_CHANGE",
                                     "MERGE_PROPOSAL", "SPLIT_PROPOSAL"],
                        },
                        "narrative": {"type": "string"},
                        "narrative_name": {"type": "string"},
                        "ticker": {"type": "string"},
                        "tickers": {"type": "array", "items": {"type": "string"}},
                        "role": {"type": "string", "enum": ["core", "secondary"]},
                        "confidence": {"type": "number"},
                        "quantitative_evidence": {"type": "object"},
                        "semantic_evidence": {"type": "string"},
                        "previous_state": {"type": "object"},
                        "proposed_state": {"type": "object"},
                        "reason": {"type": "string"},
                        "remove_reason_code": {"type": "string"},
                        "emergency": {"type": "boolean"},
                    },
                },
            },
        },
    },
}


# Full-Universe semantic classification (scripts/reconcile_narrative_universe.py,
# NOT the weekly review): assigns exactly one PRIMARY narrative and up to
# `max_secondary_narratives` SECONDARY narratives per ticker, from ONLY
# fundamental/semantic information (company name, SIC, description) plus the
# existing narrative catalog. Deliberately a separate tool schema/function
# from propose_narrative_changes above — the weekly-review Change-Set
# contract is untouched by this addition. The schema's `description` and the
# caller's system prompt both forbid momentum/RS/thrust/price-performance
# reasoning (Full-Universe spec point 16) — this call never receives that
# data in the first place, so it structurally cannot leak in.
CLASSIFY_TICKERS_TOOL_SCHEMA = {
    "name": "classify_tickers",
    "description": (
        "Classify each given ticker into exactly one PRIMARY economic narrative and "
        "optionally a small number of SECONDARY narratives, using ONLY the fundamental/"
        "semantic information provided (company name, SIC code/description, company "
        "description) plus the existing narrative catalog. NEVER reason about momentum, "
        "relative strength, thrust, or price performance — none of that is provided to "
        "you, and a classification must not depend on it."
    ),
    "input_schema": {
        "type": "object",
        "required": ["classifications"],
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["ticker", "primary_narrative_id", "primary_confidence", "reasoning"],
                    "properties": {
                        "ticker": {"type": "string"},
                        "primary_narrative_id": {
                            "type": "string",
                            "description": "Existing narrative id from the catalog. If truly none fits, "
                                            "invent a new stable snake_case id AND set primary_is_new=true "
                                            "with primary_new_name/primary_new_definition.",
                        },
                        "primary_is_new": {"type": "boolean"},
                        "primary_new_name": {"type": "string"},
                        "primary_new_definition": {
                            "type": "string",
                            "description": "Required if primary_is_new: 1-2 sentence economic/business-model definition.",
                        },
                        "primary_confidence": {"type": "number"},
                        "secondary_narratives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["narrative_id", "confidence"],
                                "properties": {
                                    "narrative_id": {"type": "string"},
                                    "is_new": {"type": "boolean"},
                                    "new_name": {"type": "string"},
                                    "new_definition": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                            },
                        },
                        "reasoning": {"type": "string"},
                    },
                },
            },
        },
    },
}


class LLMError(RuntimeError):
    pass


def _call_tool(system_prompt, user_prompt, tool_schema, tool_name, result_key, model, max_tokens):
    """Shared HTTP/tool-use plumbing behind generate_proposal_changes and
    generate_ticker_classifications — one place to get retries/error-shape
    right for every LLM-backed call in this pipeline."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY nicht gesetzt")

    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": [tool_schema],
        "tool_choice": {"type": "tool", "name": tool_name},
    }

    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=180)
    except requests.RequestException as e:
        raise LLMError(f"Anthropic API Request fehlgeschlagen: {e}")

    if resp.status_code != 200:
        raise LLMError(f"Anthropic API HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == tool_name:
            result = block.get("input", {}).get(result_key)
            if not isinstance(result, list):
                raise LLMError(f"Tool-Antwort enthält kein '{result_key}'-Array")
            return result

    raise LLMError(f"Keine tool_use-Antwort mit {tool_name} erhalten (stop_reason={data.get('stop_reason')})")


def generate_proposal_changes(system_prompt, user_prompt, model=None, max_tokens=8000):
    """Calls the configured LLM provider and returns the raw `changes` list
    from the forced tool call. Raises LLMError on any failure — callers
    must NOT fall back to treating a failed call as "no changes"; a failed
    weekly review should stop the workflow, not silently produce an empty
    (misleadingly "clean") proposal."""
    return _call_tool(system_prompt, user_prompt, CHANGE_SET_TOOL_SCHEMA,
                       "propose_narrative_changes", "changes", model, max_tokens)


def generate_ticker_classifications(system_prompt, user_prompt, model=None, max_tokens=8000):
    """Calls the configured LLM provider and returns the raw `classifications`
    list from the forced classify_tickers tool call (Full-Universe semantic
    classification — see scripts/reconcile_narrative_universe.py). Raises
    LLMError on any failure; callers must fail the batch/run rather than
    silently publishing an unclassified ticker (Full-Universe spec point 15)."""
    return _call_tool(system_prompt, user_prompt, CLASSIFY_TICKERS_TOOL_SCHEMA,
                       "classify_tickers", "classifications", model, max_tokens)


if __name__ == "__main__":
    print("Dies ist ein Modul, kein CLI-Skript. Import via review_narratives.py.", file=sys.stderr)
    sys.exit(1)
