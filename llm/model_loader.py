from config.config import (
    MODEL_NAME,
    USE_REACT_PLANNER_LORA,
    REACT_LORA_PATH,
)

def load_hf_model():
    print(f"Loading model: {MODEL_NAME}")