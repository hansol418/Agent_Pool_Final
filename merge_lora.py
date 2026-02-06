# merge_lora.py
#
# 30B base + ReAct LoRA 를 병합해서
# fp16 단일 모델로 저장하는 스크립트

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config.paths import REACT_LORA_DIR, MERGED_MODEL_DIR
from config.config import BASE_MODEL_NAME

LORA_PATH = REACT_LORA_DIR.as_posix()
OUTPUT_DIR = MERGED_MODEL_DIR.as_posix()


def main():
    print(f"[1] 토크나이저 로딩: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[2] 베이스 모델 로딩 (fp16, device_map='auto')")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"[3] LoRA 어댑터 로딩: {LORA_PATH}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
    )

    print("[4] LoRA를 base 가중치에 병합(merge_and_unload)")
    merged_model = lora_model.merge_and_unload()

    # pad_token_id / use_cache 등 설정은 그대로 유지
    if getattr(merged_model.config, "pad_token_id", None) is None:
        merged_model.config.pad_token_id = tokenizer.pad_token_id
    merged_model.config.use_cache = False

if __name__ == "__main__":
    main()