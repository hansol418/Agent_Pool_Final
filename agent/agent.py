# agent/agent.py
# HuggingFace 30B + LangGraph ReAct Agent

from typing import TypedDict
import re

from langgraph.graph import StateGraph, END

from llm.planner_inference import planner_generate
from tools.web_search import WebSearchTool
from tools.doc_search import DocSearchTool
from tools.summarize import SummarizeTool


# ------------------------------------------------
# 1) 상태 정의 (LangGraph에서 주고받는 데이터 형태)
# ------------------------------------------------
class AgentState(TypedDict, total=False):
    query: str                      # 유저 원 질문
    messages: list[str]             # LLM 중간 결과 로그 (디버깅용)
    last_action: str                # 직전에 선택된 도구 이름 또는 "FINAL"
    action_input: str               # 도구에 넘길 입력
    tool_result: str                # 마지막 도구 실행 결과
    final_answer: str               # 최종 답변 텍스트
    summarize_count: int            # summarize 도구를 몇 번 썼는지 카운트


# 도구 인스턴스 (한 번만 생성)
_web = WebSearchTool()
_doc = DocSearchTool()
_sum = SummarizeTool()


# ------------------------------------------------
# 2) LLM 노드: ReAct 스타일로 Thought / Action / Action Input 생성
# ------------------------------------------------
def agent_llm_node(state: AgentState) -> AgentState:
    """에이전트의 두뇌 역할: 지금까지의 상황을 보고
    다음에 어떤 도구를 쓸지, 또는 최종 답을 줄지 결정한다.
    """

    query = state.get("query", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])

    # 이전까지의 도구 실행 로그(Observation)를 프롬프트에 싹 모아준다.
    history_block = "\n\n".join(messages) if messages else "없음"

    # 직전 도구 결과를 Observation으로 연결
    if tool_result:
        history_block += f"\n\n[Observation]\n{tool_result}"

    prompt = f"""
당신은 도구를 사용할 수 있는 한국어 지능형 에이전트입니다.

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
{query}

위 형식을 따라 다음 행동을 결정하세요.
""".strip()

    # 🔹 ReAct Planner 전용 호출: chat 템플릿 없이 순수 프롬프트로 호출
    #    → 토큰 수를 64로 줄여서 속도 절약
    llm_output = planner_generate(
        prompt,
        max_new_tokens=160,
        temperature=0.0,  # 결정적 행동
    ).strip()

    print("\n[AGENT LLM OUTPUT]\n", llm_output, "\n")

    # 로그 저장
    messages.append(llm_output)
    state["messages"] = messages

    # 🔹 형식이 완전히 깨졌을 때 방어 로직
    #    (영어 Action: 도 없고, 한글 행동: 도 없으면 그냥 전체를 최종 답변으로 사용)
    if ("Action:" not in llm_output) and ("행동:" not in llm_output):
        state["last_action"] = "FINAL"
        state["action_input"] = llm_output
        state["final_answer"] = llm_output
        return state

    # -----------------------------
    # Thought / Action / Action Input 파싱 (한글 태그도 방어적으로 처리)
    # -----------------------------
    thought_match = re.search(r"(Thought|생각)\s*:\s*(.*)", llm_output)
    if thought_match:
        thought = thought_match.group(2).strip()
    else:
        thought = ""

    # Action 또는 행동:
    action_match = re.search(r"(Action|행동)\s*:\s*([^\n]+)", llm_output)
    raw_action = action_match.group(2).strip() if action_match else "FINAL"

    # Action Input 또는 입력:
    input_match = re.search(r"(Action Input|입력)\s*:\s*(.*)", llm_output, re.DOTALL)
    action_input = input_match.group(2).strip() if input_match else ""

    # -----------------------------
    # 액션 문자열 정규화 (web_search / doc_search / summarize / FINAL로 매핑)
    # -----------------------------
    act_lower = raw_action.lower()

    if "web_search" in act_lower or ("web" in act_lower and "search" in act_lower) or ("웹" in act_lower and "search" in act_lower):
        action = "web_search"
    elif "doc_search" in act_lower or ("doc" in act_lower and "search" in act_lower) or ("문서" in act_lower and "search" in act_lower):
        action = "doc_search"
    elif "summarize" in act_lower or "요약" in act_lower:
        action = "summarize"
    elif "final" in act_lower:
        action = "FINAL"
    else:
        # 모르는 액션이면 안전하게 FINAL 처리
        action = "FINAL"

    state["last_action"] = action
    state["action_input"] = action_input

    # -------------------------------------------------
    # Action 이 FINAL이면, Action Input 을 최종 답변으로 저장
    # + 플래너가 다시 템플릿/Thought 를 붙여버린 경우 꼬리를 잘라낸다.
    # -------------------------------------------------
    if action.upper() == "FINAL":
        # 1) 기본적으로는 Action Input 을 우선 사용
        answer = action_input.strip() or llm_output

        # 2) 플래너가 실수로 프롬프트/템플릿을 다시 붙인 경우 잘라내기
        stop_markers = [
            "\n위 형식을 따라",
            "\n위 형식을 따라 다음 행동을 결정하세요.",
            "\n위 행동을 결정하고",
            "\nThought:",
            "\nThoughts:",
            "\n생각:",
            "\nAction:",
            "\n행동:",
        ]

        for marker in stop_markers:
            idx = answer.find(marker)
            if idx != -1:
                answer = answer[:idx].strip()
                break

        state["final_answer"] = answer

    return state


# ------------------------------------------------
# 3) 각 Tool Node: last_action / action_input 기반으로 실행
#    → 속도 위해 툴 실행 후 바로 final_answer 채우고 종료
# ------------------------------------------------
def web_search_node(state: AgentState) -> AgentState:
    action_input = state.get("action_input", "").strip()
    query = action_input or state.get("query", "")

    print(f"[TOOL] web_search 실행, 입력: {query}")
    result = _web.run(query)
    state["tool_result"] = f"[web_search 결과]\n{result}"
    state["final_answer"] = state["tool_result"]
    return state


def doc_search_node(state: AgentState) -> AgentState:
    action_input = state.get("action_input", "").strip()
    query = action_input or state.get("query", "")

    print(f"[TOOL] doc_search 실행, 입력: {query}")
    result = _doc.run(query)
    state["tool_result"] = f"[doc_search 결과]\n{result}"
    state["final_answer"] = state["tool_result"]
    return state


def summarize_node(state: AgentState) -> AgentState:
    """
    summarize 도구:
    - 지금까지 messages + tool_result를 합쳐 요약
    - summarize_count 를 1 증가
    - 요약 결과를 tool_result 및 final_answer 로 기록
    """
    count = state.get("summarize_count", 0) + 1
    state["summarize_count"] = count

    text_pieces = []
    if "messages" in state:
        text_pieces.extend(state["messages"])
    if "tool_result" in state:
        text_pieces.append(state["tool_result"])

    text = "\n\n".join(text_pieces) if text_pieces else state.get("query", "")
    print(f"[TOOL] summarize 실행 (count={count})")
    result = _sum.run(text)

    state["tool_result"] = f"[summarize 결과]\n{result}"
    state["final_answer"] = result
    return state


# ------------------------------------------------
# 4) 분기 로직: 다음으로 어느 노드로 갈지 결정
# ------------------------------------------------
def decide_next_node(state: AgentState):
    action = state.get("last_action", "FINAL")
    action_lower = action.lower()
    summarize_count = state.get("summarize_count", 0)

    # summarize 반복 제한
    if action_lower == "summarize":
        if summarize_count >= 1:
            return END
        else:
            return "summarize"

    if action_lower == "web_search":
        return "web_search"
    elif action_lower == "doc_search":
        return "doc_search"
    elif action_lower == "final":
        return END
    else:
        return END


# ------------------------------------------------
# 5) LangGraph 그래프 구성
# ------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent_llm", agent_llm_node)
graph.add_node("web_search", web_search_node)
graph.add_node("doc_search", doc_search_node)
graph.add_node("summarize", summarize_node)

graph.set_entry_point("agent_llm")

graph.add_conditional_edges(
    "agent_llm",
    decide_next_node,
    {
        "web_search": "web_search",
        "doc_search": "doc_search",
        "summarize": "summarize",
        END: END,
    },
)

# 🔹 툴 → agent_llm 로 돌아가는 edge 는 제거 (툴 실행 후 바로 종료)
# graph.add_edge("web_search", "agent_llm")
# graph.add_edge("doc_search", "agent_llm")
# graph.add_edge("summarize", "agent_llm")

app = graph.compile()


# ------------------------------------------------
# 6) 외부에서 호출할 함수 (Gradio에서 사용)
# ------------------------------------------------
def run_langgraph_agent(query: str) -> str:
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "summarize_count": 0,
    }

    final_state = app.invoke(initial_state)

    answer = final_state.get("final_answer")
    if not answer:
        answer = final_state.get("tool_result", "답변을 생성하지 못했습니다.")

    return answer.strip()


# ------------------------------------------------
# 7) UI 호환용 Agent 클래스 (Wrapper)
# ------------------------------------------------
class Agent:
    def __init__(self, tools=None):
        self.tools = tools or []

    def answer(self, query: str) -> str:
        return run_langgraph_agent(query)
