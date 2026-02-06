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
    print("병합 작업 시작")


if __name__ == "__main__":
    main()