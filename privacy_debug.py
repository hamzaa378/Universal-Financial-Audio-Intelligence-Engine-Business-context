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
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def cleanup_privacy_debug_dir(base_dir: str, ttl_hours: float = 24.0) -> dict:
    """Delete stale opt-in raw debug runs.

    Raw debug transcripts intentionally contain unredacted PII. v5.1 therefore gives
    debug mode a default retention window. Cleanup is best-effort and never affects
    normal privacy processing if a file is locked or cannot be removed.
    """
    root=Path(base_dir).expanduser().resolve()
    if ttl_hours <= 0 or not root.exists():
        return {"deleted":0,"errors":0}
    cutoff=time.time()-float(ttl_hours)*3600.0
    deleted=0; errors=0
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("run_"):
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
                deleted+=1
        except Exception:
            errors+=1
    return {"deleted":deleted,"errors":errors}


def write_privacy_debug_bundle(
    raw_text: str,
    safe_text: str,
    entities: list[dict],
    recovery: dict,
    *,
    base_dir: str,
    ttl_hours: float = 24.0,
    include_raw: bool = True,
) -> dict:
    root=Path(base_dir).expanduser().resolve()
    cleanup=cleanup_privacy_debug_dir(str(root),ttl_hours=ttl_hours)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir=root/f"run_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True,exist_ok=False)

    raw_path=run_dir/"raw_asr_transcript.txt"
    safe_path=run_dir/"final_safe_transcript.txt"
    spans_path=run_dir/"redaction_spans.json"
    recovery_path=run_dir/"recovery_telemetry.json"
    comparison_path=run_dir/"comparison.json"

    if include_raw:
        raw_path.write_text(raw_text,encoding="utf-8")
        try:
            raw_path.chmod(0o600)
        except Exception:
            pass
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
        "warning":("RAW DEBUG ARTIFACT CONTAINS UNREDACTED TRANSCRIPT/PII. Delete after diagnosis."
                   if include_raw else "Raw transcript file was NOT written. Enable FINAI_PRIVACY_DEBUG_RAW=1 only for consented/synthetic diagnosis."),
    }
    comparison_path.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    return {
        "enabled":True,
        "directory":str(run_dir),
        "raw_transcript":str(raw_path) if include_raw else None,
        "safe_transcript":str(safe_path),
        "redaction_spans":str(spans_path),
        "recovery_telemetry":str(recovery_path),
        "comparison":str(comparison_path),
        "warning":comparison["warning"],
        "cleanup":cleanup,
        "ttl_hours":float(ttl_hours),
        "raw_enabled":bool(include_raw),
    }
