# v4.5 changes

## P0 privacy fixes

1. **IFSC normalization**
   - canonical `ABCD0123456`
   - hyphen/space/underscore/dot separator forms such as `ABCD-0123456`
   - spoken zero/digit forms under IFSC context
   - generic reference IDs are not auto-masked merely because separators can be removed

2. **Spoken normalization**
   - digit words + double/triple expansion
   - multi-level spoken email reconstruction
   - natural-language DOB detection under birth context

3. **NAME boundaries**
   - token-based extraction after explicit identity labels
   - stops at field transitions such as `and my account...`
   - avoids the old `field name is required` false positive

4. **ADDRESS boundaries**
   - clause-bounded capture
   - stops before phone/email/PAN/IFSC/account/OTP/CVV/UPI/DOB fields
   - residential free-form candidates are conservative and can request AI review

5. **Audio privacy alignment**
   - accepted entities receive Whisper token IDs, token times and alignment confidence
   - bleep/mute uses token ownership first
   - character-ratio mapping is fallback-only with extra safety padding

## P1 hybrid AI

- Added optional ONNX token-classification privacy NER adapter.
- NER is used only as supporting evidence on ambiguous candidates.
- Existing FastEmbed semantic privacy judge remains batched/cached.
- Strong validators bypass neural inference.

## Startup verification

`run_frontend.bat` performs a real CUDA/FP16 Whisper smoke inference before starting Streamlit.

`verify_components.bat` reports ASR GPU, semantic AI, privacy judge, optional NER and diarization separately.

## Evaluation

Added a labeled audio-manifest evaluator that separates ASR WER, text privacy accuracy, token alignment coverage and audio-redaction coverage.
