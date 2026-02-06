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