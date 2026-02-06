# tools/summarize.py

from llm.inference import llm_chat
from tools.base import Tool

class SummarizeTool(Tool):
    name = "summarize"
    description = "텍스트 요약 도구"

    def run(self, text: str) -> str:
        prompt = f"""
다음 내용을 간결하게 요약해줘. 핵심 문장 3~4문장 이내로 정리해줘.

{text}

요약:
"""
        # 🔹 내부 summarize 도 짧게 (토큰 128 제한)
        return llm_chat(prompt, max_new_tokens=128)
