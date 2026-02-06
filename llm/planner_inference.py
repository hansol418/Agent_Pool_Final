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


def planner_generate(
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
) -> str:
    """
    ReAct Planner 전용 generate 함수.

    - chat 템플릿, system prompt 등 아무 것도 안 붙이고
      "순수 텍스트 프롬프트"만 그대로 넣는다.
    - test_react_planner_only.py 에서 쓰던 패턴과 동일하게 동작하게 만드는 게 목표.
    """
    if _PLANNER_MODEL is None or _PLANNER_TOKENIZER is None:
        raise RuntimeError(
            "planner_generate: Planner model is not set. "
            "Call set_planner_model(model, tokenizer) from app.py "
            "after loading the base+LoRA model."
        )

    model = _PLANNER_MODEL
    tokenizer = _PLANNER_TOKENIZER

    model.eval()
    device = next(model.parameters()).device

    # 1) 프롬프트를 그대로 토크나이즈
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    input_length = input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 2) 프롬프트 부분 잘라내고, 새로 생성된 토큰만 디코딩
    new_tokens = generated_ids[0, input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text.strip()
