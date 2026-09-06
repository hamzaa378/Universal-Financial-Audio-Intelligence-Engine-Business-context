"""Attach privacy entities to Faster-Whisper word IDs and timestamps.

Text character spans remain useful for rendering a protected transcript. Audio privacy
uses token ownership whenever possible so punctuation/normalization differences do not
shift the bleep window.
"""
from __future__ import annotations

import re


def segment_offsets(segments: list[dict]) -> list[tuple[int,int,dict]]:
    out=[]; cursor=0
    for i,seg in enumerate(segments):
        text=str(seg.get("text", ""))
        s=cursor; e=s+len(text)
        out.append((s,e,seg))
        cursor=e+(1 if i < len(segments)-1 else 0)
    return out


def word_char_spans(segment_text: str, words: list[dict]) -> list[tuple[int,int,int,dict]]:
    """Return local char spans for words, preserving word list indices."""
    spans=[]; cursor=0; lower=segment_text.casefold()
    for wi,w in enumerate(words):
        token=str(w.get("word", "")).strip()
        if not token:
            continue
        candidates=[token]
        clean=re.sub(r"^[^\w@+]+|[^\w@.%+\-]+$", "", token, flags=re.UNICODE)
        if clean and clean != token:
            candidates.append(clean)
        idx=-1; used=""
        for cand in candidates:
            idx=lower.find(cand.casefold(), cursor)
            if idx >= 0:
                used=cand; break
        if idx < 0:
            # Search a small look-ahead before giving up. Whisper punctuation can differ.
            m=re.search(r"\S+", segment_text[cursor:])
            if not m:
                continue
            idx=cursor+m.start(); end=cursor+m.end()
        else:
            end=idx+len(used)
        spans.append((idx,end,wi,w))
        cursor=max(cursor,end)
    return spans


def global_word_map(asr: dict) -> list[dict]:
    out=[]; global_idx=0
    for seg_start,seg_end,seg in segment_offsets(asr.get("segments", [])):
        seg_text=str(seg.get("text", ""))
        for ls,le,local_i,w in word_char_spans(seg_text, seg.get("words",[]) or []):
            out.append({
                "token_id":global_idx,
                "segment_id":seg.get("id"),
                "segment_word_index":local_i,
                "char_start":seg_start+ls,
                "char_end":seg_start+le,
                "time_start":float(w.get("start",seg.get("start",0.0))),
                "time_end":float(w.get("end",seg.get("end",seg.get("start",0.0)))),
                "confidence":float(w.get("confidence",seg.get("confidence",0.0) or 0.0)),
            })
            global_idx+=1
    return out


def attach_entity_tokens(asr: dict, entities: list[dict]) -> list[dict]:
    """Return entity copies enriched with token IDs/times/alignment quality."""
    words=global_word_map(asr)
    out=[]
    for entity in entities:
        x=dict(entity)
        s=int(entity.get("start",0)); e=int(entity.get("end",s))
        hits=[w for w in words if w["char_start"] < e and s < w["char_end"]]
        if hits:
            x["token_ids"]=[w["token_id"] for w in hits]
            x["time_start"]=min(w["time_start"] for w in hits)
            x["time_end"]=max(w["time_end"] for w in hits)
            x["alignment_confidence"]=round(sum(w["confidence"] for w in hits)/len(hits),4)
            x["alignment_method"]="whisper_word_tokens"
        else:
            x["token_ids"]=[]
            x["alignment_confidence"]=0.0
            x["alignment_method"]="unresolved"
        out.append(x)
    return out
