"""Optional ONNX token-classification helper for ambiguous privacy candidates.

This is deliberately a *second opinion*, not the primary privacy detector. The default
model is an ONNX PII NER model and is downloaded only by install_ner_ai.bat. The core
application remains functional when this optional component is absent.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from functools import lru_cache

import numpy as np

_MODEL_DIR: Path | None = None
_SESSION = None
_TOKENIZER = None
_ID2LABEL: dict[int,str] = {}
_ERROR: str | None = None
_PROVIDER: str | None = None

DEFAULT_REPO = os.getenv("FINAI_NER_REPO", "onnx-community/multilang-pii-ner-ONNX")
DEFAULT_DIR = Path(os.getenv("FINAI_NER_MODEL_DIR", str(Path.home()/".cache"/"devsoc_pii_ner")))


def install_model(repo_id: str = DEFAULT_REPO, target_dir: str | Path = DEFAULT_DIR) -> str:
    """Download the smallest preferred ONNX NER artifact plus tokenizer/config files."""
    from huggingface_hub import HfApi, hf_hub_download
    target=Path(target_dir); target.mkdir(parents=True,exist_ok=True)
    files=set(HfApi().list_repo_files(repo_id))
    preferences=(
        "onnx/model_q4f16.onnx", "onnx/model_int8.onnx",
        "onnx/model_quantized.onnx", "onnx/model_uint8.onnx",
        "onnx/model_fp16.onnx", "onnx/model.onnx",
    )
    model_file=next((x for x in preferences if x in files),None)
    if model_file is None:
        raise FileNotFoundError("No supported ONNX model artifact was found in "+repo_id)
    assets=["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","sentencepiece.bpe.model"]
    for filename in [x for x in assets if x in files]+[model_file]:
        hf_hub_download(repo_id=repo_id,filename=filename,local_dir=str(target))
    (target/"selected_model.txt").write_text(model_file,encoding="utf-8")
    return str(target)


def _find_onnx(root: Path) -> Path | None:
    marker=root/"selected_model.txt"
    if marker.exists():
        p=root/marker.read_text(encoding="utf-8").strip()
        if p.exists(): return p
    for rel in ("onnx/model_q4f16.onnx","onnx/model_int8.onnx","onnx/model_quantized.onnx","onnx/model_uint8.onnx","onnx/model_fp16.onnx","onnx/model.onnx"):
        p=root/rel
        if p.exists(): return p
    matches=list(root.rglob("*.onnx"))
    return matches[0] if matches else None


def _load() -> bool:
    global _MODEL_DIR,_SESSION,_TOKENIZER,_ID2LABEL,_ERROR,_PROVIDER
    if _SESSION is not None:
        return True
    if _ERROR is not None:
        return False
    try:
        root=DEFAULT_DIR
        if not (root/"tokenizer.json").exists():
            raise FileNotFoundError(
                f"NER model is not installed at {root}. Run install_ner_ai.bat."
            )
        model_path=_find_onnx(root)
        if model_path is None:
            raise FileNotFoundError(f"No ONNX model file found under {root}")
        from tokenizers import Tokenizer
        import onnxruntime as ort
        providers=ort.get_available_providers()
        requested=os.getenv("FINAI_NER_DEVICE","auto").lower().strip()
        if requested in {"cuda","gpu"} and "CUDAExecutionProvider" not in providers:
            raise RuntimeError("NER CUDA requested but CUDAExecutionProvider is unavailable")
        use_cuda=(requested in {"cuda","gpu"}) or (requested=="auto" and "CUDAExecutionProvider" in providers)
        selected=["CUDAExecutionProvider","CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        _SESSION=ort.InferenceSession(str(model_path),providers=selected)
        _PROVIDER=_SESSION.get_providers()[0] if _SESSION.get_providers() else None
        _TOKENIZER=Tokenizer.from_file(str(root/"tokenizer.json"))
        cfg=json.loads((root/"config.json").read_text(encoding="utf-8"))
        _ID2LABEL={int(k):str(v) for k,v in (cfg.get("id2label") or {}).items()}
        _MODEL_DIR=root
        return True
    except Exception as exc:
        _ERROR=f"{type(exc).__name__}: {exc}"
        return False


def status() -> dict:
    ok=_load()
    return {
        "available":ok,
        "backend":"ONNX token classification" if ok else "disabled/unavailable",
        "provider":_PROVIDER,
        "model_repo":DEFAULT_REPO,
        "model_dir":str(DEFAULT_DIR),
        "error":_ERROR,
    }


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x=x-np.max(x,axis=-1,keepdims=True)
    e=np.exp(x)
    return e/np.sum(e,axis=-1,keepdims=True)


def _base_label(label: str) -> tuple[str,str]:
    if "-" in label and label[:2] in {"B-","I-"}:
        return label[:1],label[2:]
    return "B",label


def _privacy_type(label: str) -> str | None:
    u=label.upper().replace("-","_")
    if any(k in u for k in ("GIVENNAME","SURNAME","MIDDLENAME","FIRSTNAME","LASTNAME","PERSON","PER")):
        return "NAME"
    if any(k in u for k in ("STREET","CITY","BUILDINGNUM","BUILDING","ADDRESS","ZIP","POSTCODE")):
        return "ADDRESS"
    if any(k in u for k in ("TELEPHONE","PHONE","MOBILE")):
        return "PHONE"
    if "EMAIL" in u:
        return "EMAIL"
    if any(k in u for k in ("ACCOUNT","BANKACCOUNT")):
        return "ACCOUNT_NUMBER"
    if any(k in u for k in ("DATE","BIRTHDATE","DOB")):
        return "DOB"
    return None


@lru_cache(maxsize=256)
def predict(text: str) -> tuple[dict,...]:
    """Return grouped entity proposals with character offsets."""
    if not text.strip() or not _load():
        return tuple()
    enc=_TOKENIZER.encode(text)
    ids=np.asarray([enc.ids],dtype=np.int64)
    mask=np.asarray([enc.attention_mask],dtype=np.int64)
    feeds={}
    for inp in _SESSION.get_inputs():
        name=inp.name
        if name=="input_ids": feeds[name]=ids
        elif name=="attention_mask": feeds[name]=mask
        elif name=="token_type_ids": feeds[name]=np.zeros_like(ids)
    logits=_SESSION.run(None,feeds)[0][0]
    probs=_softmax_rows(logits)
    best=np.argmax(probs,axis=-1)
    offsets=enc.offsets
    pieces=[]
    for i,label_id in enumerate(best):
        s,e=offsets[i]
        if e<=s:
            continue
        label=_ID2LABEL.get(int(label_id),str(label_id))
        if label.upper()=="O":
            continue
        prefix,base=_base_label(label)
        ptype=_privacy_type(base)
        if not ptype:
            continue
        pieces.append({"start":int(s),"end":int(e),"label":base,"type":ptype,"confidence":float(probs[i,int(label_id)]),"prefix":prefix})
    if not pieces:
        return tuple()
    grouped=[]
    for p in pieces:
        if grouped and p["type"]==grouped[-1]["type"] and p["start"] <= grouped[-1]["end"]+1 and p["prefix"]!="B":
            g=grouped[-1]; g["end"]=p["end"]; g["confidence"]=(g["confidence"]+p["confidence"])/2
        else:
            grouped.append(dict(p))
    return tuple({k:v for k,v in g.items() if k!="prefix"} for g in grouped)


def _clause_bounds(text: str, start: int, end: int, radius: int=140) -> tuple[int,int]:
    lo=max(0,start-radius); hi=min(len(text),end+radius)
    for sep in (".","!","?","\n",";"):
        pos=text.rfind(sep,lo,start)
        if pos>=lo: lo=max(lo,pos+1)
        pos2=text.find(sep,end,hi)
        if pos2>=0: hi=min(hi,pos2)
    return lo,hi


def support_candidates(text: str, items: list[dict]) -> list[dict]:
    """Return NER overlap support for candidate items, one clause at a time."""
    if not items or not _load():
        return [{"available":False,"support":0.0,"overlaps":[]} for _ in items]
    cache={}
    out=[]
    for item in items:
        lo,hi=_clause_bounds(text,int(item["start"]),int(item["end"]))
        clause=text[lo:hi]
        if clause not in cache:
            cache[clause]=list(predict(clause))
        relevant=[]
        for p in cache[clause]:
            gs=lo+p["start"]; ge=lo+p["end"]
            # Accept exact type or the NAME/ADDRESS family for ambiguous contextual fields.
            type_ok=(p["type"]==item.get("type")) or ({p["type"],item.get("type")} <= {"NAME","ADDRESS"})
            if type_ok and gs < int(item["end"]) and int(item["start"]) < ge:
                relevant.append({**p,"start":gs,"end":ge})
        score=max((float(x["confidence"]) for x in relevant),default=0.0)
        out.append({"available":True,"support":round(score,4),"overlaps":relevant})
    return out
