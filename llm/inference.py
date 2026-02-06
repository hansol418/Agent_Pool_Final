import unicodedata
import torch

from config.config import TEMPERATURE, MAX_NEW_TOKENS
from llm.model_loader import load_hf_model


# ✅ 1) 베이스 + LoRA 모델 한 번 로드
tokenizer, model = load_hf_model()


def llm_chat(prompt: str) -> str:
    with torch.no_grad():
        output_ids = model.generate()
    
    return ""