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

    # -------------------------------------------------
    # 2) 베이스 모델 (fp16, device_map='auto')
    #    - 여기서는 bitsandbytes 사용 안 함 (순수 fp16)
    # -------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # generate() 시 warning 줄이기 위해 pad_token_id 지정
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA + SFT 학습 시에는 use_cache=False 가 더 안전한 경우가 많음
    if hasattr(model, "config"):
        model.config.use_cache = False

    print("[✅] Base model loaded (fp16, no quantization)")