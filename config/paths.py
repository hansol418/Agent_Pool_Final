# config/paths.py
from pathlib import Path

# ✅ 레포 루트: llm_Rag_Chatbot_Agent_Testing
REPO_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = REPO_ROOT / "assets"
MODELS_DIR = REPO_ROOT / "models"
SFT_DIR    = REPO_ROOT / "llm_sft"

# 너 프로젝트에서 쓰는 기본 폴더들(권장 이름)
MERGED_MODEL_DIR = MODELS_DIR / "merged_react_30b"
REACT_LORA_DIR   = MODELS_DIR / "react_planner_lora"

# 학습 데이터/결과(실행 위치 영향 제거용)
REACT_DATA_JSONL = SFT_DIR / "react_sft_data.jsonl"
REACT_LORA_OUT   = REACT_LORA_DIR  # 학습 결과를 models 쪽에 두는 방식
