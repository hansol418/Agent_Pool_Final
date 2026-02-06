# tools/web_search.py
"""
웹 검색 도구 (Serper 기반)

- SERPER_API_KEY 환경변수를 사용해서 https://google.serper.dev/search 호출
- 상위 3개 결과만 뽑아서 title + snippet + URL 형태로 압축
- 전체 길이가 너무 길면 잘라서 반환 (LLM 프롬프트 폭주 방지)
"""

import os
import textwrap
from typing import Optional

import requests


MAX_WEB_RESULT_CHARS = 1200  # web_search 결과 문자열 최대 길이


class WebSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        # 환경변수에서 Serper API 키를 가져옴
        self.api_key = api_key or os.getenv("SERPER_API_KEY")

    def _build_error(self, msg: str) -> str:
        return f"[web_search] {msg}"

    def run(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return self._build_error("검색 질의가 비어 있습니다.")

        if not self.api_key:
            return self._build_error(
                "SERPER_API_KEY 가 설정되어 있지 않습니다. 환경변수에 API 키를 추가해주세요."
            )

        print(f"[TOOL] web_search (Serper) 실행, 입력: {query}")

        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": 5,  # 최대 5개 정도만 받아놓고, 실제로는 상위 3개만 사용
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return self._build_error(f"Serper 요청 중 오류가 발생했습니다: {e}")

        organic = data.get("organic", [])[:3]
        if not organic:
            return self._build_error(f"'{query}' 에 대한 검색 결과를 찾지 못했습니다.")

        # 상위 3개 결과를 간단한 텍스트로 정리
        lines = []
        for idx, item in enumerate(organic, 1):
            title = (item.get("title") or "").strip()
            snippet = (
                item.get("snippet")
                or item.get("description")
                or ""
            ).strip()
            link = (item.get("link") or "").strip()

            # snippet 이 너무 길면 2줄 정도로 자르기
            snippet = snippet.replace("\n", " ")
            snippet = textwrap.shorten(snippet, width=220, placeholder="…")

            block = f"[{idx}] {title}\n요약: {snippet}"
            if link:
                block += f"\nURL: {link}"
            lines.append(block)

        result_text = "\n\n".join(lines)

        # LLM 프롬프트 부담 줄이기 위해 결과도 일정 길이 이상이면 잘라냄
        if len(result_text) > MAX_WEB_RESULT_CHARS:
            result_text = result_text[:MAX_WEB_RESULT_CHARS] + "…"

        return result_text
