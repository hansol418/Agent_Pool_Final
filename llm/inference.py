import unicodedata
import torch

from config.config import TEMPERATURE, MAX_NEW_TOKENS
from llm.model_loader import load_hf_model

# 🔹 ReAct Planner용 전역 모델 세팅 함수 가져오기
from llm.planner_inference import set_planner_model


# ✅ 1) 베이스 + LoRA 모델 한 번 로드
tokenizer, model = load_hf_model()

# ✅ 2) pad_token 설정 (기존 그대로 유지)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 3) ReAct Planner 쪽에서도 동일 모델/토크나이저를 사용하도록 연결
#    - 이렇게 하면 planner_generate()가 이 모델을 그대로 사용
set_planner_model(model, tokenizer)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    cleaned = []
    for ch in text:
        if ch == "\ufffd":
            continue
        if ch == "\n" or ch.isprintable():
            cleaned.append(ch)
    return "".join(cleaned)


def llm_chat(prompt: str, max_new_tokens: int | None = None) -> str:
    """
    prompt -> reply
    max_new_tokens 를 주면 그 값 사용, 아니면 config 의 MAX_NEW_TOKENS 사용.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    new_tokens = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # prompt 부분 제거
    if full_text.startswith(prompt):
        reply = full_text[len(prompt):].strip()
    else:
        reply = full_text.strip()

    reply = _clean_text(reply)
    return reply
