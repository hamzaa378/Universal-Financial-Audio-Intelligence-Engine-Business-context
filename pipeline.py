from ingest.audio_loader import load_audio
from ingest.call_quality import call_quality
from ingest.tamper_detection import detect_tamper
from asr.fintech_asr import transcribe
from nlp.language import annotate_segment_languages
from nlp.pii import detect_pii, mask_pii
from nlp.intent import classify_intent
from nlp.entities import extract_entities
from nlp.obligation import detect_obligations
from nlp.regulatory import check_regulatory
from analysis_ai.emotion import detect_stress_markers
from analysis_ai.diarization import diarize, assign_speakers

def _trust(asr_conf, quality, intent_conf):
    return round(max(0,min(1,0.55*asr_conf+0.25*quality+0.20*intent_conf)),3)

def run_pipeline(audio_path, model_name=None):
    audio,sr=load_audio(audio_path)
    quality=call_quality(audio,sr)
    asr=transcribe(audio_path, model_name)
    asr["segments"]=annotate_segment_languages(asr["segments"])
    dia=diarize(audio_path)
    if dia.get("turns"): asr["segments"]=assign_speakers(asr["segments"],dia["turns"])
    text=asr["text"]
    pii=detect_pii(text)
    intent=classify_intent(text)
    entities=extract_entities(text)
    obligations=detect_obligations(text)
    result={
      "audio":{"quality":quality,"tamper_screen":detect_tamper(audio,sr),"diarization":dia},
      "transcription":asr,
      "understanding":{
        "intent":intent,"financial_entities":entities,"obligations":obligations,
        "pii":pii,"regulatory":check_regulatory(text),"stress_markers":detect_stress_markers(audio,sr)
      },
      "privacy":{"masked_transcript":mask_pii(text,pii)},
      "confidence":{"overall_trust":_trust(asr["confidence"],quality["score"],intent["confidence"]),
                    "interpretation":"Composite triage score; retain component confidences for audit."}
    }
    return result
