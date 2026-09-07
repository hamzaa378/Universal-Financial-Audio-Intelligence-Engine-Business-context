"""Opt-in privacy debugging artifacts for v4.11.

Raw ASR text can contain sensitive information, so this module is disabled by default.
When explicitly enabled it writes the primary ASR transcript, final safe transcript,
accepted span metadata, recovery telemetry and a small comparison manifest.  This lets
us tell whether text disappeared in ASR or during redaction without silently logging PII
in normal operation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_privacy_debug_bundle(
    raw_text: str,
    safe_text: str,
    entities: list[dict],
    recovery: dict,
    *,
    base_dir: str,
) -> dict:
    root=Path(base_dir).expanduser().resolve()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir=root/f"run_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True,exist_ok=False)

    raw_path=run_dir/"raw_asr_transcript.txt"
    safe_path=run_dir/"final_safe_transcript.txt"
    spans_path=run_dir/"redaction_spans.json"
    recovery_path=run_dir/"recovery_telemetry.json"
    comparison_path=run_dir/"comparison.json"

    raw_path.write_text(raw_text,encoding="utf-8")
    safe_path.write_text(safe_text,encoding="utf-8")
    # Strip any internal raw alternate-ASR content if a future implementation adds it.
    public_entities=[]
    for e in entities:
        public_entities.append({k:v for k,v in dict(e).items() if k not in {"alternate_text","raw_alternate_text"}})
    spans_path.write_text(json.dumps(public_entities,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    recovery_path.write_text(json.dumps(recovery or {},indent=2,ensure_ascii=False,default=str),encoding="utf-8")

    comparison={
        "raw_length":len(raw_text),
        "safe_length":len(safe_text),
        "character_delta":len(raw_text)-len(safe_text),
        "accepted_redaction_spans":len(public_entities),
        "raw_sha256":_sha256_text(raw_text),
        "safe_sha256":_sha256_text(safe_text),
        "warning":"RAW DEBUG ARTIFACT CONTAINS UNREDACTED TRANSCRIPT/PII. Delete after diagnosis.",
    }
    comparison_path.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    return {
        "enabled":True,
        "directory":str(run_dir),
        "raw_transcript":str(raw_path),
        "safe_transcript":str(safe_path),
        "redaction_spans":str(spans_path),
        "recovery_telemetry":str(recovery_path),
        "comparison":str(comparison_path),
        "warning":comparison["warning"],
    }
