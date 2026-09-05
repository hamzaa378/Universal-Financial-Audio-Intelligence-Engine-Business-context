# Universal Financial Audio Intelligence Engine — AI v2

This version upgrades the original prototype from hard-coded placeholders to a confidence-aware hybrid AI pipeline.

## What changed
- Fintech-conditioned Faster-Whisper ASR with beam search, VAD and word timestamps.
- `small` is the default model; use `medium` or `large-v3` when accuracy matters more than latency/VRAM.
- Segment-level language annotation for code-switched calls.
- Optional multilingual NLI intent classifier (`FINAI_ZERO_SHOT=1`) with deterministic fallback.
- Optional pyannote Community-1 speaker diarization (`FINAI_DIARIZATION=1`).
- Financial entity extraction, promise-to-pay detection, PII masking and regulatory screening.
- Real acoustic quality metrics and conservative tamper/replay screening.
- Stress *markers* rather than unsafe definitive emotion claims.
- Every layer reports its own confidence; an overall trust score is provided only for triage.

## Setup (recommended clean venv on Windows)
```powershell
cd Devsoc_AI_v2
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Core mode does **not** require torchvision or torchaudio.

Run:
```powershell
python app.py sample_call.wav --model small --output result.json
```

For higher ASR accuracy on an RTX 4060, try:
```powershell
python app.py sample_call.wav --model medium
```
If CUDA/CTranslate2 runtime libraries are unavailable the code automatically retries on CPU/int8.

## Optional AI intent model
Install `transformers` and `sentencepiece`, then set:
```powershell
$env:FINAI_ZERO_SHOT="1"
```
The configured multilingual NLI model will classify intents without a fixed trained classifier. For a final product, fine-tune a multilingual financial intent model on your labeled call corpus.

## Optional diarization
Install `pyannote.audio`, accept the model conditions on Hugging Face, set `HF_TOKEN`, then:
```powershell
$env:FINAI_DIARIZATION="1"
```

## Accuracy roadmap
1. Build a labeled evaluation corpus of real/synthetic fintech calls with English, Hindi and Hinglish.
2. Measure WER/CER, entity F1, intent macro-F1, PTP precision/recall, PII F1 and diarization DER.
3. Use hard examples to fine-tune ASR/intent/NER rather than adding generic model size blindly.
4. Keep regulatory phrase rules jurisdiction/version controlled and human-approved.
5. Calibrate confidence against held-out data before using thresholds operationally.

## Important limitation
Tamper, replay, stress and regulatory outputs are screening signals. They must not be presented as definitive forensic, psychological or legal determinations.
