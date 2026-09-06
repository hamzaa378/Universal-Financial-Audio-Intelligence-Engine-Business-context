# Universal Financial Audio Intelligence Engine — v4.4 GPU-First

v4.4 is a Windows-focused financial-call intelligence prototype with strict GPU ASR, hybrid PII/privacy decisions, protected-audio generation, financial entity extraction, promise-to-pay detection, regulatory phrase screening, and a Streamlit review UI.

## Main architecture

```text
Upload / microphone
   -> one audio decode (16 kHz mono)
   -> shared acoustic quality/stress analysis
   -> Faster-Whisper on CUDA FP16
   -> time-aligned words + confidence
   -> PII candidates + strong validators
        -> ambiguous candidates -> batched semantic AI
   -> sensitive financial-ID detector
   -> profanity detector
   -> financial entity / intent / obligation / compliance layer
   -> protected transcript
   -> word-time aligned protected audio (beep or mute)
   -> Streamlit review UI + benchmark reports
```

## Strict GPU behavior

Normal startup uses:

```text
FINAI_DEVICE=cuda
FINAI_COMPUTE_TYPE=float16
FINAI_STRICT_GPU=1
```

If CUDA inference fails, v4.4 reports the error and stops that request. It does **not** silently re-run Faster-Whisper on CPU. Use `run_frontend_cpu.bat` only when you intentionally want CPU troubleshooting.

## Install

```bat
install_v4.bat
```

Core install includes Faster-Whisper, audio processing, Streamlit, and CPU FastEmbed semantic AI. Optional heavier diarization is in `requirements-ai.txt`.

Optional semantic-AI GPU acceleration:

```bat
install_semantic_gpu.bat
```

## Verify GPU

Open a new Command Prompt after changing PATH and check:

```bat
where cublas64_12.dll
where cublasLt64_12.dll
where cudnn64_9.dll
```

Then:

```bat
diagnose_gpu.bat
gpu_smoke_test.bat
```

## Start frontend

```bat
run_frontend.bat
```

Open `http://localhost:8501` and click **Warm up AI models** once before the first call.

Recommended RTX 4060 demo setup:

- Whisper: `small` + `Fast demo` for lowest latency
- Whisper: `medium` + `Balanced` for stronger accuracy
- Detection: `Balanced hybrid`
- Semantic AI: ON after warm-up
- Full PII redaction: ON
- Sensitive financial IDs: ON
- Profanity reduction: ON
- Protected audio: Bleep tone
- Speaker diarization: OFF unless pyannote + HF token are configured

## Privacy categories

PII:
- Email
- Phone
- PAN
- IFSC
- UPI/VPA
- Card number (Luhn validated)
- Aadhaar (checksum/context aware)
- Bank account number
- OTP
- CVV
- PIN code
- Date of birth
- Name (explicit context)
- Address (explicit/contextual)
- Passport
- Voter ID / EPIC
- Driving licence/license

Sensitive financial identifiers are tracked separately:
- Transaction/reference ID
- Loan ID
- Customer ID
- Application ID
- Complaint/grievance/case ID

## Financial understanding

The current layer extracts or screens:
- EMI amount
- Outstanding amount
- Due amount
- Loan/principal amount
- Settlement amount
- Late fee / bounce charge
- Interest rate / APR
- Tenure
- Due/payment date expressions
- Payment method
- Payment status
- Promise-to-pay
- Repayment difficulty
- Payment refusal
- Payment/debt dispute
- Complaint/fraud/verification/foreclosure/settlement intents
- Abuse/profanity markers
- Regulatory risk phrases and selected disclosures
- Acoustic stress markers
- Optional speaker diarization

## Efficiency changes in v4.4

- ASR receives the already-decoded waveform instead of decoding the file again.
- Call-quality and stress-marker features share one acoustic feature pass.
- Audio masking can reuse the loaded waveform.
- Semantic AI batches all missing prototype embeddings and caches them.
- Intent + abusive tone share one semantic inference batch.
- Ambiguous PII candidates share one semantic privacy-judge batch.
- Pyannote pipeline is cached when diarization is enabled.
- UI reports ASR time, text-AI time, total time, and real-time factor (RTF).

## Benchmarks

```bat
run_privacy_benchmark.bat
```

It runs:
1. unit tests,
2. PII/profanity precision-recall-FPR regression,
3. sensitive-financial-ID regression,
4. optional semantic-AI stress test,
5. optional full GPU audio benchmark.

Audio folder example:

```bat
run_privacy_benchmark.bat "C:\path\to\test_audio" small
```

Reports are written under `reports\`.

The bundled test data is synthetic regression data. Do not present its scores as real-world production accuracy. For a defensible evaluation, add consented/labeled or carefully generated noisy financial calls and report WER/CER, PII precision/recall/F1, negative-case FPR, profanity F1, intent/entity F1, diarization error rate, and end-to-end RTF.
