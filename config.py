from dataclasses import dataclass, field
import os

@dataclass
class Settings:
    whisper_model: str = os.getenv("FINAI_WHISPER_MODEL", "small")
    device: str = os.getenv("FINAI_DEVICE", "auto")
    compute_type: str = os.getenv("FINAI_COMPUTE_TYPE", "auto")
    beam_size: int = int(os.getenv("FINAI_BEAM_SIZE", "5"))
    best_of: int = int(os.getenv("FINAI_BEST_OF", "5"))
    vad_filter: bool = True
    enable_diarization: bool = os.getenv("FINAI_DIARIZATION", "0") == "1"
    hf_token: str | None = os.getenv("HF_TOKEN")
    enable_zero_shot: bool = os.getenv("FINAI_ZERO_SHOT", "0") == "1"
    nli_model: str = os.getenv("FINAI_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    financial_vocab: list[str] = field(default_factory=lambda: [
        "EMI", "APR", "KYC", "e-KYC", "NPA", "NBFC", "RBI", "SEBI", "CIBIL",
        "foreclosure", "prepayment", "outstanding balance", "overdue", "principal",
        "interest rate", "late fee", "loan account", "disbursement", "moratorium",
        "Aadhaar", "PAN", "IFSC", "UPI", "NEFT", "RTGS", "IMPS", "beneficiary",
        "promise to pay", "PTP", "grievance", "collection", "settlement"
    ])

SETTINGS = Settings()
