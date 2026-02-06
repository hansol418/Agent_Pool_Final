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
    
if __name__ == "__main__":
    main()