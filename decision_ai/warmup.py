from __future__ import annotations

from decision_ai.semantic_engine import warmup, status
from decision_ai.privacy_judge import prototype_texts as privacy_texts
from decision_ai.utterance_ai import prototype_texts as utterance_texts


def warmup_semantic_ai() -> dict:
    texts=list(dict.fromkeys(privacy_texts()+utterance_texts()))
    result=warmup(texts)
    result["prototype_count"]=len(texts)
    return result


def semantic_status() -> dict:
    return status()
