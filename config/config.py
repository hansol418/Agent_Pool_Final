# config/config.py
# ================================================================
# 📘 전역 환경 설정
# ================================================================

import os
from config.paths import MERGED_MODEL_DIR, REACT_LORA_DIR

# ✅ 베이스 LLM 모델 (현재 에이전트가 쓰는 30B 모델)
BASE_MODEL_NAME = "davidkim205/komt-llama-30b-v1"
MODEL_NAME = os.getenv("MODEL_NAME", MERGED_MODEL_DIR.as_posix())

# ✅ 생성 관련 설정
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# ✅ RAG / 임베딩 관련 설정
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ✅ Gradio 서버 설정
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860

# ✅ Sentence-Transformers 임베딩 모델
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ================================================================
# 🔧 ReAct Planner LoRA 설정
#   - 우리가 SFT로 학습한 LoRA 어댑터를 쓸지 여부
#   - 저장된 LoRA 디렉토리 경로
# ================================================================

# LoRA 사용 여부 (True 로 두면 아래 REACT_LORA_PATH 를 시도해서 로딩)
# USE_REACT_PLANNER_LORA = True

# LoRA 디렉토리 경로(SFT튜닝진행)
# 👉 실제 학습 환경 기준 경로에 맞게 수정해도 됨.
#    (지금은 llm_sft 프로젝트 위치 기준 절대경로 예시)
# REACT_LORA_PATH = "/home/super/llm_sft/apps/llm_sft/react_planner_lora"

USE_REACT_PLANNER_LORA = False   # 이미 병합했으니 LoRA 다시 붙이지 않음
REACT_LORA_PATH = os.getenv("REACT_LORA_PATH", REACT_LORA_DIR.as_posix()) # 지금은 사용 안됨
