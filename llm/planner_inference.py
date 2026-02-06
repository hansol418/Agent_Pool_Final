# llm/planner_inference.py
# ReAct Planner 전용 모델 호출 모듈
#
# - app.py 에서 이미 로드해서 LoRA까지 붙인 모델/토크나이저를
#   여기로 넘겨서(global) ReAct 용으로만 generate 를 호출하는 용도.
#
# - test_react_planner_only.py 에서 했던 방식과 최대한 비슷하게,
#   "프롬프트 그대로" 넣고 generate -> 프롬프트 길이 잘라내고 생성 부분만 사용.

from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# 전역 모델/토크나이저 (app.py 에서 세팅해 줄 것)
_PLANNER_MODEL: Optional[PreTrainedModel] = None
_PLANNER_TOKENIZER: Optional[PreTrainedTokenizerBase] = None


def set_planner_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
    """
    app.py 에서 base 모델 + LoRA 를 다 붙인 뒤에 호출하는 함수.

    예)
        from llm.planner_inference import set_planner_model
        model, tokenizer = load_base_model_and_lora(...)
        set_planner_model(model, tokenizer)
    """
    global _PLANNER_MODEL, _PLANNER_TOKENIZER
    _PLANNER_MODEL = model
    _PLANNER_TOKENIZER = tokenizer


