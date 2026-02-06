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


# -------------------------------------------------------
# 3) 모델 / 토크나이저 로드 (bf16 LoRA, no bitsandbytes)
# -------------------------------------------------------

def load_model_and_tokenizer():
    print(f"[🔄] Loading base model (bf16, no bitsandbytes): {BASE_MODEL_NAME}")

    # 3-1) 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=False,
    )

    # pad_token 없으면 eos_token으로 맞춰주기
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SFT에서는 일반적으로 right padding 사용
    tokenizer.padding_side = "right"

    # 3-2) 모델 로드 (bfloat16, 양자화 없음)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,   # ✅ bf16
        device_map="auto",            # 여러 GPU 있으면 자동 분산
    )

    # LoRA 학습 시에는 use_cache=False 로 두는 게 안정적
    if hasattr(model, "config"):
        model.config.use_cache = False

    print("[✅] Model & tokenizer loaded (bf16, no quantization)")
    return model, tokenizer


# -------------------------------------------------------
# 4) LoRA 설정
# -------------------------------------------------------

def make_lora_config():
    """
    Planner 전용 LoRA 설정.
    너무 aggressive 하지 않게 적당한 r, alpha 사용.
    """
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return lora_config


# -------------------------------------------------------
# 5) TrainingArguments 설정
# -------------------------------------------------------

def make_training_args(output_dir: str) -> TrainingArguments:
    """
    A100 80GB 기준, 가벼운 SFT 예시용 설정.
    (실제 데이터가 늘어나면 epoch/step 조정 가능)
    """
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,      # 30B 기준 안전한 배치사이즈
        gradient_accumulation_steps=16,     # effective batch size ~= 16
        num_train_epochs=3,                 # 데이터 크기에 따라 조정
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        # ✅ fp16 스케일링 끄고 bf16만 사용
        fp16=False,
        bf16=True,

        gradient_checkpointing=True,

        # bitsandbytes 8bit 옵티마이저 대신, 표준 AdamW 사용
        optim="adamw_torch",

        report_to=[],
    )
    return args


# -------------------------------------------------------
# 6) main: 전체 학습 파이프라인
# -------------------------------------------------------

def main():
    # 6-1) Dataset 로드
    print("[📁] Loading dataset...")
    train_dataset = load_react_dataset(DATA_PATH)
    print(f"[📁] Train samples: {len(train_dataset)}")

    # 6-2) 모델 / 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()

    # 6-3) LoRA 설정
    lora_config = make_lora_config()

    # 6-4) TrainingArguments 설정
    training_args = make_training_args(OUTPUT_DIR)

    # 6-5) SFTTrainer 생성
    print("[🚀] Starting SFT training (ReAct Planner)...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,   # prompt+response를 하나의 텍스트로 변환
        max_seq_length=1024,               # ReAct 프롬프트+응답 길이 여유 있게
        args=training_args,
    )

    # 6-6) 학습 시작
    trainer.train()

    # 6-7) LoRA 어댑터 저장
    trainer.save_model()
    print(f"[✅] LoRA planner model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
