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