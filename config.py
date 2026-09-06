from dataclasses import dataclass, field
import os


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    whisper_model: str = os.getenv("FINAI_WHISPER_MODEL", "small")
    device: str = os.getenv("FINAI_DEVICE", "auto")
    compute_type: str = os.getenv("FINAI_COMPUTE_TYPE", "auto")
    strict_gpu: bool = _env_bool("FINAI_STRICT_GPU", False)
    cuda_index: int = int(os.getenv("FINAI_CUDA_INDEX", "0"))
    beam_size: int = int(os.getenv("FINAI_BEAM_SIZE", "2"))
    best_of: int = int(os.getenv("FINAI_BEST_OF", "1"))
    vad_filter: bool = _env_bool("FINAI_VAD", True)
    enable_diarization: bool = _env_bool("FINAI_DIARIZATION", False)
    hf_token: str | None = os.getenv("HF_TOKEN")
    enable_zero_shot: bool = _env_bool("FINAI_ZERO_SHOT", False)
    nli_model: str = os.getenv("FINAI_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    enable_semantic_ai: bool = _env_bool("FINAI_SEMANTIC_AI", False)
    semantic_device: str = os.getenv("FINAI_SEMANTIC_DEVICE", "auto")
    privacy_profile: str = os.getenv("FINAI_PRIVACY_PROFILE", "balanced")
    repetition_penalty: float = float(os.getenv("FINAI_REPETITION_PENALTY", "1.05"))
    no_repeat_ngram_size: int = int(os.getenv("FINAI_NO_REPEAT_NGRAM", "3"))
    hallucination_silence_threshold: float = float(os.getenv("FINAI_HALLUCINATION_SILENCE", "1.4"))
    financial_vocab: list[str] = field(default_factory=lambda: [
        "EMI", "APR", "KYC", "e-KYC", "NPA", "NBFC", "RBI", "SEBI", "CIBIL",
        "foreclosure", "prepayment", "outstanding balance", "overdue", "principal",
        "interest rate", "late fee", "bounce charge", "loan account", "disbursement",
        "moratorium", "Aadhaar", "PAN", "IFSC", "IFSC code", "bank IFSC", "UPI", "VPA", "NEFT", "RTGS",
        "IMPS", "beneficiary", "promise to pay", "PTP", "grievance", "collection",
        "settlement", "waiver", "repossession", "autodebit", "NACH", "e-mandate",
        "transaction reference", "loan ID", "customer ID", "passport", "voter ID",
        "driving licence", "driving license", "account number", "bank account number", "OTP", "one time password", "CVV", "PIN code", "postal code", "date of birth"
    ])

SETTINGS = Settings()
