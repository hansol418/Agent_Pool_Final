# make_react_dataset.py
import json
from pathlib import Path
import random

# 1) 공통 프롬프트 템플릿 (agent_llm_node 프롬프트 축약 버전)
BASE_PROMPT_TEMPLATE = """당신은 도구를 사용할 수 있는 한국어 지능형 에이전트입니다.

사용 가능한 도구:
- web_search : 웹 검색을 통해 최신 정보, 일반 상식, 생활 정보를 찾습니다.
- doc_search : 내부 문서(AI 개념, 자연어 처리, 강화학습 등)를 검색합니다.
- summarize  : 여러 정보를 요약하여 정리합니다.
- FINAL      : 지금까지의 정보를 바탕으로 최종 답변을 바로 제공합니다.

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


def make_sample(history_block: str, user_query: str, thought: str, action: str, action_input: str):
    """하나의 SFT 샘플(dict)을 만들어주는 헬퍼 함수"""
    prompt = BASE_PROMPT_TEMPLATE.format(
        history_block=history_block,
        user_query=user_query,
    )
    response = f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"
    return {"prompt": prompt, "response": response}


# ------------------------------------------------
# 1) web_search 샘플 (최신/동향/시장/뉴스)
# ------------------------------------------------
def build_web_search_samples(target_count: int = 80):
    """최신/동향/시장/뉴스 → web_search 선택 샘플들"""
    base_user_queries = [
        "2025년 기준 자율주행 자동차의 최신 동향을 알려줘",
        "2025년 인공지능 산업의 주요 트렌드가 뭐야?",
        "최근 한국의 반도체 시장 동향이 궁금해",
        "요즘 챗GPT 같은 LLM 서비스 비즈니스 모델이 어떻게 발전하고 있어?",
        "2025년 기준으로 전기차 시장 점유율을 알려줘",
        "최신 머신러닝 논문 트렌드를 알고 싶어",
        "최근 사이버 보안 위협 동향을 정리해줘",
        "요즘 클라우드 비용 최적화 트렌드가 뭐야?",
        "최근 게임 업계에서 AI가 어떻게 활용되고 있는지 알려줘",
        "2025년 기준으로 주목받는 오픈소스 LLM 프로젝트가 뭐가 있어?",
    ]
    thought_templates = [
        "최신 동향과 수치는 웹에서 정보를 찾는 것이 가장 적절하다.",
        "실시간에 가까운 산업 정보는 웹 검색을 통해 모으는 것이 좋다.",
        "시장 동향과 트렌드는 내부 문서보다 웹 검색이 더 최신이다.",
        "업계 트렌드는 변화가 빠르므로 웹 검색 도구를 이용해야 한다.",
    ]

    samples = []
    # target_count 만큼 랜덤하게 생성 (중복 허용)
    while len(samples) < target_count:
        q = random.choice(base_user_queries)
        t = random.choice(thought_templates)

        # Action Input 을 살짝 변형
        action_input = q.replace("알려줘", "").replace("궁금해", "").strip()
        if not action_input:
            action_input = q

        samples.append(
            make_sample(
                history_block="없음",
                user_query=q,
                thought=t,
                action="web_search",
                action_input=action_input,
            )
        )

    return samples


# ------------------------------------------------
# 2) doc_search 샘플 (개념/정의/기초 설명)
# ------------------------------------------------
def build_doc_search_samples(target_count: int = 100):
    """개념/정의/기초 설명 → doc_search 선택 샘플들"""

    # (1) 핵심 개념 고정 페어 → 여러 번 보게 해서 패턴을 강하게 주입
    fixed_pairs = [
        ("강화학습이 뭐야? 직관적으로 설명해줘", "강화학습의 정의와 직관적인 설명"),
        ("자연어 처리가 뭐야?", "자연어 처리의 개념과 예시"),
        ("인공지능(AI)의 기본 개념을 정리해줘", "인공지능(AI)의 정의와 기본 개념 정리"),
        ("LLM이 뭐야? 쉽게 설명해줘", "LLM(대규모 언어모델)의 개념과 특징"),
        ("벡터 임베딩이 뭐야?", "벡터 임베딩의 정의와 직관적 설명"),
        ("RAG 구조가 뭐고, 언제 쓰는지 알려줘", "RAG의 개념과 동작 방식 설명"),
    ]

    base_user_queries = [
        "강화학습이 뭐야? 직관적으로 설명해줘",
        "자연어 처리가 정확히 뭐야?",
        "인공지능(AI)의 기본 개념을 정리해줘",
        "지도학습과 비지도학습의 차이를 알려줘",
        "딥러닝이 머신러닝과 어떻게 다른지 설명해줘",
        "트랜스포머 모델의 핵심 아이디어를 간단히 알려줘",
        "LLM이 뭐야? 쉽게 설명해줘",
        "벡터 임베딩이라는 개념을 이해하기 쉽게 설명해줘",
        "RAG 구조가 뭐고, 언제 쓰는지 알려줘",
        "온톨로지라는 개념이 인공지능에서 어떻게 쓰이는지 알려줘",
    ]

    thought_templates = [
        "이 질문은 내부에 정리된 AI 개념 문서를 참고하는 것이 가장 적절하다.",
        "기본 개념은 웹 검색보다 내부 문서(doc)를 참조하는 것이 일관성이 있다.",
        "핵심 이론과 정의는 내부 문서에 잘 정리되어 있으므로 doc_search가 알맞다.",
        "학습 개념 관련 질문이므로 내부 AI 문서를 검색해야 한다.",
    ]

    samples = []

    # 1차로 고정 페어들을 한 번씩 넣어준다
    for q, ai in fixed_pairs:
        t = random.choice(thought_templates)
        samples.append(
            make_sample(
                history_block="없음",
                user_query=q,
                thought=t,
                action="doc_search",
                action_input=ai,
            )
        )

    # 나머지는 base_user_queries에서 랜덤하게 채운다
    def infer_action_input(query: str) -> str:
        if "강화학습" in query:
            return "강화학습의 정의와 직관적인 설명"
        elif "자연어 처리" in query:
            return "자연어 처리의 개념과 예시"
        elif "지도학습과 비지도학습" in query:
            return "지도학습과 비지도학습의 차이 정리"
        elif "트랜스포머" in query:
            return "트랜스포머 모델 핵심 아이디어 요약"
        elif "LLM" in query:
            return "LLM(대규모 언어모델)의 개념과 특징"
        elif "벡터 임베딩" in query:
            return "벡터 임베딩의 정의와 직관적 설명"
        elif "RAG" in query:
            return "RAG의 개념과 동작 방식 설명"
        elif "온톨로지" in query:
            return "온톨로지의 개념과 AI에서의 활용 예시"
        elif "인공지능(AI)" in query or "AI의 기본 개념" in query:
            return "인공지능(AI)의 정의와 기본 개념 정리"
        else:
            return "해당 개념의 정의와 직관적인 설명"

    while len(samples) < target_count:
        q = random.choice(base_user_queries)
        t = random.choice(thought_templates)
        ai = infer_action_input(q)

        samples.append(
            make_sample(
                history_block="없음",
                user_query=q,
                thought=t,
                action="doc_search",
                action_input=ai,
            )
        )

    return samples


# ------------------------------------------------
# 3) summarize 샘플 (이미 web_search 결과가 있는 상태)
# ------------------------------------------------
def build_summarize_samples(target_count: int = 50):
    """이미 web_search 결과(Observation)가 있는 상태에서 summarize 선택 샘플들"""
    web_obs_1 = """[web_search 결과]
[1] 2025년 8월, 테슬라 최신 혁신 동향 및 산업 OS 변화 ...
[2] 2025 자율주행 기술 동향 및 시장분석 ...
[3] 2025 년 국내외 자율주행차 기술, 시장 전망과 사업화 전략 ..."""

    web_obs_2 = """[web_search 결과]
[1] 2025년 인공지능 산업 투자 규모 및 주요 기업 동향 ...
[2] LLM 기반 서비스 사례와 수익모델 분석 ...
[3] 주요 국가별 AI 규제 및 정책 동향 ..."""

    web_obs_list = [web_obs_1, web_obs_2]

    base_user_queries = [
        "위 검색 결과를 바탕으로 2025년 자율주행 자동차 기술 동향을 한 번에 정리해줘",
        "위에 나온 내용들로 2025년 인공지능 산업 주요 특징을 요약해줘",
        "검색된 결과를 기반으로 핵심 포인트만 bullet 형식으로 정리해줘",
        "자율주행차와 관련된 기술/시장/정책 동향을 한 문단으로 정리해줘",
        "LLM 비즈니스 모델에 대한 주요 인사이트를 간단히 요약해줘",
    ]
    thought_templates = [
        "웹 검색을 통해 충분한 정보를 모았으니 이제 요약해서 정리하면 된다.",
        "이미 Observation에 필요한 정보가 모였으므로 summarize 도구로 핵심만 추리면 된다.",
        "추가 검색보다, 현재 결과를 기반으로 요약하는 것이 효율적이다.",
    ]

    samples = []
    while len(samples) < target_count:
        q = random.choice(base_user_queries)
        t = random.choice(thought_templates)
        obs = random.choice(web_obs_list)

        ai = q.replace("정리해줘", "요약").replace("요약해줘", "요약").strip()
        if not ai:
            ai = "위 검색 결과의 핵심 내용 요약"

        samples.append(
            make_sample(
                history_block=obs,
                user_query=q,
                thought=t,
                action="summarize",
                action_input=ai,
            )
        )

    return samples


# ------------------------------------------------
# 4) FINAL 샘플 (도구 없이 바로 답하는 케이스)
# ------------------------------------------------
def build_final_samples(target_count: int = 50):
    """도구 없이 바로 FINAL 로 답하는 케이스들"""
    base_user_queries = [
        "너는 어떤 역할을 하는 에이전트야?",
        "간단하게 인사 한 번만 해줘",
        "도구 말고 너 자신에 대해서 짧게 소개해줘",
        "이 대화 시스템이 어떤 구조로 동작하는지 아주 간단히 말해줘",
        "지금은 그냥 잡담만 하고 싶어. 부담 없는 인사말 해줘",
    ]
    thought_templates = [
        "이 질문은 도구가 필요 없으므로 바로 답변하면 된다.",
        "웹 검색이나 문서 검색 없이도 충분히 답변할 수 있는 내용이다.",
        "시스템 프롬프트에 기반해서 바로 답변하는 것이 자연스럽다.",
    ]

    samples = []
    while len(samples) < target_count:
        q = random.choice(base_user_queries)
        t = random.choice(thought_templates)

        if "역할" in q:
            ai = "나는 한국어로 대화하면서 web_search, doc_search, summarize 같은 도구를 활용해 정보를 찾아주고 정리해 주는 지능형 에이전트야."
        elif "인사" in q:
            ai = "안녕하세요! 저는 여러 도구를 활용해서 정보를 찾아주고 정리해 주는 한국어 지능형 에이전트입니다. 편하게 질문해 주세요."
        elif "구조" in q:
            ai = "나는 사용자의 질문을 받고, 필요하면 web_search나 doc_search 같은 도구를 사용해서 정보를 수집한 뒤, summarize로 정리해서 답변해 주는 에이전트야."
        else:
            ai = "안녕하세요! 오늘은 가볍게 대화하면서 궁금한 점이 있으면 언제든지 물어봐 주세요."

        samples.append(
            make_sample(
                history_block="없음",
                user_query=q,
                thought=t,
                action="FINAL",
                action_input=ai,
            )
        )

    return samples


# ------------------------------------------------
# 5) 전체 Dataset 빌드
# ------------------------------------------------
def build_dataset():
    samples = []
    # 비율 조정: web 80 / doc 100 / summarize 50 / final 50
    samples.extend(build_web_search_samples(target_count=80))
    samples.extend(build_doc_search_samples(target_count=100))
    samples.extend(build_summarize_samples(target_count=50))
    samples.extend(build_final_samples(target_count=50))

    random.shuffle(samples)
    return samples


def save_jsonl(samples, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            line = json.dumps(sample, ensure_ascii=False)
            f.write(line + "\n")


if __name__ == "__main__":
    out_path = Path("react_sft_data.jsonl")
    data = build_dataset()
    save_jsonl(data, out_path)
    print(f"✅ {len(data)}개 샘플을 {out_path} 에 저장했습니다.")
