from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import streamlit as st

from pipeline import run_pipeline, analyze_text
from asr.fintech_asr import warmup_model, backend_status
from decision_ai.warmup import warmup_semantic_ai
from decision_ai.startup import verify_startup

PII_TYPES=["EMAIL","PHONE","PAN","IFSC","UPI","CARD","AADHAAR","ACCOUNT_NUMBER","OTP","CVV","PINCODE","DOB","NAME","ADDRESS","PASSPORT","VOTER_ID","DRIVING_LICENSE","PASSWORD","USERNAME","API_KEY","AUTH_TOKEN"]

st.set_page_config(page_title="Financial Audio Intelligence", page_icon="🎙️", layout="wide")
st.title("Universal Financial Audio Intelligence Engine")
st.caption("Upload or record a call → send it to the backend → transcribe, detect, mask and listen to a protected audio copy")

with st.sidebar:
    st.header("Privacy policy")
    profile_label=st.selectbox("Detection profile", ["Balanced hybrid","High precision","High recall"], index=0)
    profile={"Balanced hybrid":"balanced","High precision":"high_precision","High recall":"high_recall"}[profile_label]
    semantic_ai=st.toggle(
        "Semantic AI second opinion", value=True,
        help="Uses lightweight semantic AI only for ambiguous PII/intent decisions. Falls back to deterministic rules if unavailable.",
    )
    ner_ai=st.toggle(
        "Token NER second opinion (optional)", value=False,
        help="Optional ONNX token-classification model for ambiguous NAME/ADDRESS/PHONE/account contexts. Install with install_ner_ai.bat.",
    )
    full_mask=st.toggle("Full PII redaction", value=True, help="Recommended. Partial mode can reveal identifier suffixes.")
    profanity_enabled=st.toggle("Reduce profanity", value=True)
    enable_diarization=st.toggle(
        "Speaker diarization (optional)", value=False,
        help="Uses pyannote when requirements-ai.txt and HF_TOKEN are configured. The model is cached after first load."
    )
    financial_ids_enabled=st.toggle(
        "Hide operational/customer IDs", value=False,
        help="Optional. Masks transaction, loan, customer, application and complaint/ticket IDs separately from core PII. Disabled by default to reduce false positives."
    )
    mask_types=set(st.multiselect("Privacy types to hide", PII_TYPES, default=PII_TYPES))
    whisper_model=st.selectbox("Whisper model", ["small","medium","large-v3"], index=0)
    speed_label=st.selectbox(
        "ASR processing mode",
        ["Fast demo","Balanced","Maximum accuracy"],
        index=0,
        help="Fast demo uses greedy decoding. Balanced uses a small beam. Maximum accuracy is much slower."
    )
    asr_speed_mode={"Fast demo":"fast","Balanced":"balanced","Maximum accuracy":"accuracy"}[speed_label]

    if st.button("Warm up & verify AI", width="stretch"):
        with st.spinner("Verifying GPU ASR, semantic judge, optional NER and diarization... "):
            try:
                report=verify_startup(whisper_model,semantic=semantic_ai,ner=ner_ai,asr_smoke=True)
                st.session_state["startup_report"]=report
                a=report.get("asr_gpu",{})
                if a.get("ready"):
                    st.success(f"ASR GPU = {str(a.get('device')).upper()} {a.get('compute_type')} ✓")
                else:
                    st.error("ASR GPU is not ready: "+str(a.get("error")))
                sem=report.get("semantic_ai",{})
                st.caption("Semantic AI = "+((str(sem.get("provider") or sem.get("backend"))+" ✓") if sem.get("ready") else "disabled/unavailable"))
                judge=report.get("privacy_judge",{})
                st.caption("Privacy Judge = "+("ready ✓" if judge.get("ready") else str(judge.get("state","fallback"))))
                ner=report.get("ner_ai",{})
                st.caption("NER AI = "+((str(ner.get("provider") or ner.get("backend"))+" ✓") if ner.get("ready") else "disabled/unavailable"))
                dia=report.get("diarization",{})
                st.caption("Diarization = "+("available ✓" if dia.get("available") else "available/disabled" if dia.get("installed") else "not installed"))
            except Exception as exc:
                st.error(f"AI verification failed: {exc}")

    bstat=backend_status()
    if bstat.get("error"):
        st.error("GPU backend error: " + str(bstat["error"]))
    else:
        planned=str(bstat.get("planned_device","?")).upper()
        st.caption(f"Planned ASR backend: {planned} / {bstat.get('planned_compute_type','?')}")
        if bstat.get("strict_gpu") and planned=="CUDA":
            st.success("Strict GPU mode: ON — no silent CPU fallback")
        elif planned=="CPU":
            st.warning("ASR is currently planned for CPU.")

    audio_method_label=st.selectbox("Protected-audio masking", ["Bleep tone","Mute sensitive audio"], index=0)
    audio_method={"Bleep tone":"beep","Mute sensitive audio":"mute"}[audio_method_label]
    st.caption("Balanced + medium Whisper is a good first demo setting. Large-v3 is more accurate but heavier.")


def options():
    return dict(
        privacy_profile=profile,
        use_semantic_ai=semantic_ai,
        use_ner_ai=ner_ai,
        mask_mode="full" if full_mask else "partial",
        mask_types=mask_types,
        profanity_enabled=profanity_enabled,
        financial_ids_enabled=financial_ids_enabled,
    )


def show_analysis(a: dict):
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("PII", a.get("pii_count",0))
    c2.metric("Sensitive IDs", a.get("sensitive_financial_id_count",0))
    c3.metric("Profanity", a.get("profanity_count",0))
    c4.metric("PII confidence", f"{100*a.get('pii_mean_confidence',1.0):.1f}%")
    c5.metric("Text analysis", f"{a.get('timing_ms',{}).get('total',0):.1f} ms")

    st.subheader("Protected transcript")
    st.text_area("Safe text", a.get("safe_text",""), height=150, disabled=True, label_visibility="collapsed")

    left,right=st.columns(2)
    with left:
        st.subheader("PII decisions")
        if a.get("pii"):
            st.dataframe(a["pii"], width="stretch", hide_index=True)
        else:
            st.info("No PII detected at the selected threshold.")
    with right:
        st.subheader("Understanding")
        st.write("**Intent:**", a.get("intent",{}).get("label"), f"({a.get('intent',{}).get('confidence',0):.2f})")
        st.write("**Abusive-tone AI:**", a.get("abusive_tone",{}))
        if a.get("financial_entities"):
            st.write("**Financial entities:**", a["financial_entities"])
        if a.get("obligations"):
            st.write("**Obligations / promises:**", a["obligations"])
        if a.get("sensitive_financial_ids"):
            st.write("**Sensitive financial IDs:**", a["sensitive_financial_ids"])
        if a.get("profanity"):
            st.write("**Profanity metadata:**", a["profanity"])
        if a.get("regulatory"):
            st.write("**Regulatory screening:**", a["regulatory"])

    ai=a.get("semantic_ai",{})
    if semantic_ai:
        if ai.get("available"):
            st.success(f"Semantic AI active: {ai.get('model')} via {ai.get('backend')}")
        else:
            st.warning("Semantic AI requested but unavailable; deterministic fallback is active. Install requirements-ai-lite.txt and run once with internet access to cache the model.")
            if ai.get("error"):
                st.caption(ai["error"])
    nai=a.get("ner_ai",{})
    if ner_ai:
        if nai.get("available"):
            st.success(f"Token NER active via {nai.get('provider') or nai.get('backend')}")
        else:
            st.warning("NER AI requested but unavailable; contextual rules + semantic judge remain active. Run install_ner_ai.bat to enable it.")
            if nai.get("error"): st.caption(nai["error"])


def _analysis_view(r: dict) -> dict:
    return {
        "safe_text":r["privacy"]["safe_transcript"],
        "pii":r["understanding"]["pii"],
        "profanity":r["understanding"]["profanity"],
        "sensitive_financial_ids":r["understanding"].get("sensitive_financial_ids",[]),
        "financial_entities":r["understanding"].get("financial_entities",[]),
        "regulatory":r["understanding"].get("regulatory",{}),
        "intent":r["understanding"]["intent"],
        "obligations":r["understanding"]["obligations"],
        "abusive_tone":r["understanding"].get("abusive_tone",{}),
        "pii_count":r["privacy"]["pii_count"],
        "sensitive_financial_id_count":r["privacy"].get("sensitive_financial_id_count",0),
        "profanity_count":r["privacy"]["profanity_count"],
        "pii_mean_confidence":r["confidence"].get("pii_mean",1.0),
        "timing_ms":{"total":r["performance"].get("privacy_and_nlp",0)},
        "semantic_ai":r["privacy"].get("semantic_ai",{}),
        "ner_ai":r["privacy"].get("ner_ai",{}),
    }


def _json_safe_result(r: dict) -> dict:
    out=deepcopy(r)
    pa=out.get("privacy",{}).get("protected_audio")
    if isinstance(pa,dict):
        pa.pop("path",None)
    return out


def _process_audio_bytes(data: bytes, suffix: str, source_name: str):
    tmp_path=None
    protected_path=None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path=tmp.name
        with st.spinner("Backend processing: ASR → AI decisions → PII/profanity masking → protected audio..."):
            r=run_pipeline(
                tmp_path,
                whisper_model,
                include_raw=False,
                create_audio_output=True,
                audio_redaction_method=audio_method,
                asr_speed_mode=asr_speed_mode,
                enable_diarization=enable_diarization,
                **options(),
            )
        pa=r.get("privacy",{}).get("protected_audio") or {}
        protected_path=pa.get("path")
        protected_bytes=Path(protected_path).read_bytes() if protected_path and Path(protected_path).exists() else b""
        r_safe=_json_safe_result(r)
        st.session_state["last_audio_result"]={
            "source_name":source_name,
            "result":r_safe,
            "protected_audio_bytes":protected_bytes,
        }
    finally:
        for p in (tmp_path,protected_path):
            if p:
                try: Path(p).unlink(missing_ok=True)
                except Exception: pass


def _show_last_audio_result():
    saved=st.session_state.get("last_audio_result")
    if not saved:
        return
    r=saved["result"]
    st.divider()
    st.success(f"Backend processing complete — source: {saved['source_name']}")

    st.subheader("Protected output audio")
    st.caption("Use the player's ▶ / ⏸ control to play or pause. Sensitive PII/profanity intervals are bleeped or muted according to the sidebar setting.")
    protected=saved.get("protected_audio_bytes",b"")
    if protected:
        st.audio(protected, format="audio/wav")
        st.download_button(
            "Download protected audio",
            data=protected,
            file_name="protected_financial_call.wav",
            mime="audio/wav",
            width="stretch",
        )
    else:
        st.warning("A protected audio file could not be generated. Transcript analysis is still available below.")

    pa=r.get("privacy",{}).get("protected_audio") or {}
    if pa:
        cols=st.columns(3)
        cols[0].metric("Audio redactions", pa.get("interval_count",0))
        cols[1].metric("Protected duration", f"{pa.get('duration_s',0):.2f} s")
        cols[2].metric("Mask method", str(pa.get("method","-")).title())
        with st.expander("Protected-audio intervals"):
            intervals=pa.get("intervals",[])
            if intervals:
                st.dataframe(intervals,width="stretch",hide_index=True)
            else:
                st.info("No sensitive audio intervals were detected; the protected copy contains no bleep/mute sections.")

    show_analysis(_analysis_view(r))

    st.subheader("Call confidence and performance")
    cc1,cc2,cc3,cc4,cc5,cc6=st.columns(6)
    cc1.metric("Overall trust", f"{100*r['confidence']['overall_trust']:.1f}%")
    cc2.metric("ASR confidence", f"{100*r['confidence']['asr']:.1f}%")
    cc3.metric("ASR", f"{r['performance'].get('asr',0)/1000:.2f} s")
    cc4.metric("Text AI", f"{r['performance'].get('privacy_and_nlp',0):.0f} ms")
    cc5.metric("Total", f"{r['performance'].get('total',0)/1000:.2f} s")
    rtf=r['performance'].get('real_time_factor')
    cc6.metric("RTF", f"{rtf:.2f}" if isinstance(rtf,(int,float)) else "-")
    backend=r.get("transcription",{}).get("backend",{})
    if backend:
        st.caption(
            f"ASR backend used: {str(backend.get('device','?')).upper()} / "
            f"{backend.get('compute_type','?')} / {backend.get('speed_mode','?')}"
        )
        if backend.get("fallback_reason"):
            st.warning("GPU fallback: " + str(backend["fallback_reason"]))
    with st.expander("Signal / compliance details"):
        st.json({
            "audio":r["audio"],
            "regulatory":r["understanding"]["regulatory"],
            "stress_markers":r["understanding"]["stress_markers"],
            "performance_ms":r["performance"],
        })
    st.download_button(
        "Download safe analysis JSON",
        json.dumps(r,indent=2,ensure_ascii=False),
        file_name="financial_audio_analysis_safe.json",
        mime="application/json",
        width="stretch",
    )


tab_audio,tab_text,tab_bench=st.tabs(["Audio processing","Text privacy playground","Benchmark"])

with tab_audio:
    st.subheader("1. Give audio to the frontend")
    source_mode=st.radio(
        "Audio input source",
        ["Upload audio file","Record from microphone"],
        horizontal=True,
        help="Both inputs are sent to the same backend pipeline.",
    )

    source_bytes=None
    source_suffix=".wav"
    source_name=""
    source_mime="audio/wav"

    if source_mode == "Upload audio file":
        uploaded=st.file_uploader(
            "Choose a custom audio or video call",
            type=["wav","mp3","m4a","flac","ogg","aac","webm","mp4","mov"],
            help="Examples: your own WAV/MP3 recording, tele-KYC call, support call or collection-call sample.",
        )
        if uploaded is not None:
            source_bytes=uploaded.getvalue()
            source_suffix=Path(uploaded.name).suffix.lower() or ".wav"
            source_name=uploaded.name
            source_mime=uploaded.type or "audio/wav"
    else:
        st.caption("Click the microphone control, record the call/sample, then stop the recording.")
        recorded=st.audio_input("Record audio directly in the frontend")
        if recorded is not None:
            source_bytes=recorded.getvalue()
            source_suffix=".wav"
            source_name="microphone_recording.wav"
            source_mime="audio/wav"

    if source_bytes:
        st.subheader("2. Preview the input")
        st.audio(source_bytes, format=source_mime)
        c1,c2=st.columns([3,1])
        with c1:
            process_clicked=st.button(
                "Send to backend & protect",
                type="primary",
                width="stretch",
                help="Runs transcription, AI privacy decisions, PII/profanity masking and creates a protected audio copy.",
            )
        with c2:
            if st.button("Clear previous output",width="stretch"):
                st.session_state.pop("last_audio_result",None)
                st.rerun()
        if process_clicked:
            try:
                _process_audio_bytes(source_bytes,source_suffix,source_name)
            except Exception as exc:
                st.error(f"Backend processing failed: {exc}")
                st.exception(exc)
    else:
        st.info("Upload a file or record from the microphone to enable backend processing.")

    _show_last_audio_result()

with tab_text:
    sample="My mobile number is 9876543210 and my PAN is ABCDE1234F. I already paid yesterday but this is shit service."
    text=st.text_area("Paste or type a transcript", value=sample, height=160)
    if st.button("Test privacy decisions", width="stretch"):
        with st.spinner("Evaluating transcript..."):
            a=analyze_text(text, **options())
        show_analysis(a)

with tab_bench:
    st.subheader("Latest local benchmark")
    report=Path("reports/privacy_benchmark.json")
    if report.exists():
        data=json.loads(report.read_text(encoding="utf-8"))
        p=data.get("pii",{}); e=data.get("text_privacy_efficiency",{})
        c1,c2,c3,c4=st.columns(4)
        c1.metric("PII precision", f"{100*p.get('precision',0):.2f}%")
        c2.metric("PII recall", f"{100*p.get('recall',0):.2f}%")
        c3.metric("Negative-case FPR", f"{100*p.get('negative_case_false_positive_rate',0):.2f}%")
        c4.metric("P95 privacy latency", f"{e.get('p95_ms',0):.3f} ms")
        st.caption(data.get("dataset",{}).get("note",""))
        with st.expander("Full benchmark JSON"):
            st.json(data)
    else:
        st.info("Run run_privacy_benchmark.bat first to generate a benchmark report.")
