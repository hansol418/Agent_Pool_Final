# tools/doc_search.py

import torch
from sentence_transformers import util

from embeddings.embedding_loader import load_embeddings
from llm.inference import llm_chat
from tools.base import Tool

# 간단한 내부 문서 예시
internal_docs = [
    {"title": "AI 개념", "content": "인공지능(AI)은 인간처럼 학습하고 사고하는 기술입니다."},
    {"title": "자연어 처리", "content": "자연어 처리는 인간의 언어를 컴퓨터가 이해하도록 하는 기술입니다."},
    {"title": "강화학습", "content": "강화학습은 보상을 최대화하는 방식으로 학습하는 알고리즘입니다."}
]

embedder = load_embeddings()
doc_embeddings = embedder.encode([d["content"] for d in internal_docs], convert_to_tensor=True)


class DocSearchTool(Tool):
    name = "doc_search"
    description = "내부 문서에서 유사한 내용을 검색"

    def run(self, query: str) -> str:
        # 1) 쿼리 임베딩 → 가장 유사한 내부 문서 선택
        q_emb = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, doc_embeddings)[0]
        top_idx = int(torch.argmax(cos_scores))

        doc = internal_docs[top_idx]

        # 2) 요약 프롬프트 (2~3문장으로 짧게)
        prompt = f"""
다음 문서를 읽고 핵심 내용을 2~3문장으로 간단히 요약해줘.

문서 내용:
{doc['content']}

요약:
""".strip()

        # 3) 30B 모델 호출 (토큰 수는 64로 제한)
        raw = llm_chat(prompt, max_new_tokens=64)
        answer = raw.strip()

        # 4) 플래너 LoRA가 붙이는 꼬리 텍스트 제거
        stop_markers = [
            "\n사용 가능한 문법",
            "\n위 행동을 결정하고",
            "\n위 형식을 따라",
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

        return f"{doc['title']}: {answer}"
