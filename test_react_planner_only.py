# test_react_planner_only.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config.paths import REACT_LORA_DIR
from config.config import BASE_MODEL_NAME


LORA_PATH = REACT_LORA_DIR.as_posix()

# ✅ 여기 템플릿은 make_react_dataset.py 의 BASE_PROMPT_TEMPLATE 와 완전히 동일하게 맞춰야 함
BASE_PROMPT_TEMPLATE = """당신은 도구를 사용할 수 있는 한국어 지능형 에이전트입니다.

사용 가능한 도구:
- web_search : 웹 검색을 통해 최신 정보, 일반 상식, 생활 정보를 찾습니다.
- doc_search : 내부 문서(AI 개념, 자연어 처리, 강화학습 등)를 검색합니다.
- summarize  : 여러 정보를 요약하여 정리합니다.

출력 형식(반드시 이 3줄만 포함):

Thought: (짧은 생각)
Action: web_search | doc_search | summarize | FINAL
Action Input: (도구에 넘길 입력 내용 또는 FINAL일 경우 최종 답변 전체)

지금까지의 대화 및 도구 결과:
{history_block}

사용자의 질문:
{user_query}

위 형식을 따라 다음 행동을 결정하세요.
"""


def build_prompt(user_query: str, history_block: str = "없음") -> str:
    return BASE_PROMPT_TEMPLATE.format(
        history_block=history_block,
        user_query=user_query,
    )


def main():
    print("[🔄] Loading base model + LoRA (ReAct planner)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 베이스 모델 로드 (fp16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # ReAct LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    # ✅ 테스트용 질문 (doc_search, web_search, FINAL 다 여러 개 시도해봐도 좋음)
    user_query = "너는 어떤 역할을 하는 에이전트야?"
    prompt = build_prompt(user_query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,   # 우선 greedy로 형식 여부만 확인
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("===== FULL OUTPUT =====")
    print(full_text)

    # 프롬프트 부분 잘라내고, 모델이 새로 생성한 부분만 보고 싶으면:
    print("\n===== GENERATED PART ONLY =====")
    print(full_text[len(prompt):])


if __name__ == "__main__":
    main()
