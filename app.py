import argparse,json
from pathlib import Path
from pipeline import run_pipeline


def main():
    p=argparse.ArgumentParser(description="Universal Financial Audio Intelligence Engine v4.6")
    p.add_argument("audio",help="Audio/video file")
    p.add_argument("--model",default=None,help="Whisper model, e.g. small, medium, large-v3")
    p.add_argument("--speed",choices=["fast","balanced","accuracy"],default="fast")
    p.add_argument("--output",default="result.json")
    p.add_argument("--privacy-profile",choices=["high_precision","balanced","high_recall"],default="balanced")
    p.add_argument("--semantic-ai",action="store_true",help="Enable batched FastEmbed semantic decisions")
    p.add_argument("--ner-ai",action="store_true",help="Enable optional ONNX token NER second opinion")
    p.add_argument("--partial-mask",action="store_true")
    p.add_argument("--no-profanity-filter",action="store_true")
    p.add_argument("--financial-id-mask",dest="financial_id_mask",action="store_true",help="Opt in to masking transaction/customer/application/ticket IDs in addition to core PII")
    p.add_argument("--no-financial-id-mask",dest="financial_id_mask",action="store_false",help=argparse.SUPPRESS)
    p.set_defaults(financial_id_mask=False)
    p.add_argument("--diarization",action="store_true")
    p.add_argument("--protected-audio",action="store_true")
    p.add_argument("--include-raw",action="store_true",help="Debug only: include raw transcript")
    a=p.parse_args()
    result=run_pipeline(
        a.audio,a.model,include_raw=a.include_raw,
        privacy_profile=a.privacy_profile,use_semantic_ai=a.semantic_ai,use_ner_ai=a.ner_ai,
        mask_mode="partial" if a.partial_mask else "full",
        profanity_enabled=not a.no_profanity_filter,
        financial_ids_enabled=a.financial_id_mask,
        create_audio_output=a.protected_audio,
        asr_speed_mode=a.speed,
        enable_diarization=a.diarization,
    )
    Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({
        "safe_transcript":result["privacy"]["safe_transcript"],
        "intent":result["understanding"]["intent"],
        "pii_count":result["privacy"]["pii_count"],
        "sensitive_financial_id_count":result["privacy"].get("sensitive_financial_id_count",0),
        "profanity_count":result["privacy"]["profanity_count"],
        "asr_backend":result["transcription"].get("backend",{}),
        "trust":result["confidence"],
        "performance":result["performance"],
        "saved":a.output,
    },indent=2,ensure_ascii=False))

if __name__=="__main__": main()
