# train_react_planner.py
# ReAct Planner SFT (LoRA, bf16 / no bitsandbytes)

import os
from dataclasses import dataclass
from typing import Dict, Optional, List
from config.paths import REACT_DATA_JSONL, REACT_LORA_OUT
from config.config import BASE_MODEL_NAME

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# -------------------------------------------------------
# 0) 기본 설정
# -------------------------------------------------------

# 학습 데이터 파일 (2단계에서 만든 JSONL)
DATA_PATH = REACT_DATA_JSONL.as_posix()

# LoRA 결과 저장 경로
OUTPUT_DIR = REACT_LORA_OUT.as_posix()


# -------------------------------------------------------
# 1) Dataset 로드
#   - JSONL: {"prompt": "...", "response": "..."} 구조
# -------------------------------------------------------

def load_react_dataset(path: str):
    """
    react_sft_data.jsonl 을 datasets 포맷으로 로드.
    """
    dataset = load_dataset("json", data_files=path)
    # dataset["train"] 하나만 쓰면 충분 (train / test 나누고 싶으면 split 가능)
    return dataset["train"]


# -------------------------------------------------------
# 2) 텍스트 포맷팅 함수
#    - prompt + response 를 하나의 텍스트로 합친다.
#    - 추론 때는 "prompt만 넣고 -> 모델이 response(3줄)를 이어서 생성"하게 사용.
# -------------------------------------------------------

def formatting_func(batch: Dict[str, List[str]]) -> List[str]:
    """
    batch["prompt"], batch["response"] 를 받아서
    "prompt + response" 형태의 단일 텍스트 시퀀스로 합친다.
    """
    outputs = []
    prompts = batch["prompt"]
    responses = batch["response"]

    for p, r in zip(prompts, responses):
        # 중간에 구분 줄 하나 정도만 넣고 바로 response를 붙인다.
        text = p.rstrip() + "\n" + r.strip()
        outputs.append(text)

    return outputs


def main():
    train_dataset = load_react_dataset(REACT_DATA_JSONL.as_posix())
    print(f"Loaded dataset size: {len(train_dataset)}")


if __name__ == "__main__":
    main()