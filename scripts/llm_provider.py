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


class LLMError(RuntimeError):
    pass


def generate_proposal_changes(system_prompt, user_prompt, model=None, max_tokens=8000):
    """Calls the configured LLM provider and returns the raw `changes` list
    from the forced tool call. Raises LLMError on any failure — callers
    must NOT fall back to treating a failed call as "no changes"; a failed
    weekly review should stop the workflow, not silently produce an empty
    (misleadingly "clean") proposal."""
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
        "tools": [CHANGE_SET_TOOL_SCHEMA],
        "tool_choice": {"type": "tool", "name": "propose_narrative_changes"},
    }

    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=180)
    except requests.RequestException as e:
        raise LLMError(f"Anthropic API Request fehlgeschlagen: {e}")

    if resp.status_code != 200:
        raise LLMError(f"Anthropic API HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "propose_narrative_changes":
            changes = block.get("input", {}).get("changes")
            if not isinstance(changes, list):
                raise LLMError("Tool-Antwort enthält kein 'changes'-Array")
            return changes

    raise LLMError("Keine tool_use-Antwort mit propose_narrative_changes erhalten "
                    f"(stop_reason={data.get('stop_reason')})")


if __name__ == "__main__":
    print("Dies ist ein Modul, kein CLI-Skript. Import via review_narratives.py.", file=sys.stderr)
    sys.exit(1)
