# llm/model_loader.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config.config import (
    MODEL_NAME,
    USE_REACT_PLANNER_LORA,
    REACT_LORA_PATH,
)

def load_hf_model():
    """
    HuggingFace LLaMA 기반 30B 모델을 로드하고,
    필요하다면 ReAct Planner용 LoRA 어댑터를 붙여서 반환한다.
    """

    print(f"[🔄] Loading HuggingFace base model: {MODEL_NAME}")

    # -------------------------------------------------
    # 1) 토크나이저 로딩
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    # pad_token 없으면 eos_token으로 맞추기 (30B 계열에서 자주 필요한 설정)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"