"""Component-level startup verification for v4.5."""
from __future__ import annotations

import os
from asr.fintech_asr import backend_status, warmup_model
from decision_ai.warmup import warmup_semantic_ai
from decision_ai.ner_engine import status as ner_status


def diarization_status() -> dict:
    token=bool(os.getenv("HF_TOKEN"))
    try:
        import pyannote.audio  # noqa: F401
        installed=True
    except Exception as exc:
        installed=False
        err=f"{type(exc).__name__}: {exc}"
    else:
        err=None
    return {
        "available":bool(installed and token),
        "installed":installed,
        "hf_token":token,
        "state":"ready" if installed and token else "disabled/unavailable",
        "error":err if not installed else (None if token else "HF_TOKEN not configured"),
    }


def verify_startup(model_name: str="small", *, semantic: bool=True, ner: bool=False, asr_smoke: bool=True) -> dict:
    report={}
    try:
        warm=warmup_model(model_name,smoke_test=asr_smoke)
        report["asr_gpu"]={"ready":warm.get("device")=="cuda","device":warm.get("device"),"compute_type":warm.get("compute_type"),"error":None}
    except Exception as exc:
        report["asr_gpu"]={"ready":False,"device":backend_status().get("planned_device"),"compute_type":backend_status().get("planned_compute_type"),"error":str(exc)}
    if semantic:
        try:
            sem=warmup_semantic_ai()
            report["semantic_ai"]={"ready":bool(sem.get("available")),**sem}
            report["privacy_judge"]={"ready":bool(sem.get("available")),"state":"ready" if sem.get("available") else "rules fallback","error":sem.get("error")}
        except Exception as exc:
            report["semantic_ai"]={"ready":False,"error":str(exc)}
            report["privacy_judge"]={"ready":False,"state":"rules fallback","error":str(exc)}
    else:
        report["semantic_ai"]={"ready":False,"state":"disabled"}
        report["privacy_judge"]={"ready":False,"state":"disabled"}
    ns=ner_status() if ner else {"available":False,"backend":"disabled","error":None}
    report["ner_ai"]={"ready":bool(ns.get("available")),**ns}
    report["diarization"]=diarization_status()
    return report
