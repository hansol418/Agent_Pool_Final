# ui/gradio_ui.py
from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr
from tools.web_search import WebSearchTool
from tools.doc_search import DocSearchTool
from tools.summarize import SummarizeTool
from agent.agent import Agent
from config.paths import ASSETS_DIR

# LangGraph 래퍼 에이전트
agent = Agent([WebSearchTool(), DocSearchTool(), SummarizeTool()])

# ✅ 절대 경로 고정 (직접 확인)
ICON_FILE = ASSETS_DIR / "agent_pool_icon.png"


def chat_fn(message, history):
    """
    message: 사용자가 막 입력한 한 줄(str)
    history: 지금까지의 대화 (list[dict], 각 dict는 {'role': ..., 'content': ...})
    """
    history = history or []

    # 1) 사용자 메시지 추가
    history.append({"role": "user", "content": message})

    # 2) 에이전트에게 질의 (현재는 마지막 user message만 사용)
    answer = agent.answer(message)

    # 3) 에이전트(assistant) 응답 추가
    history.append({"role": "assistant", "content": answer})

    # Gradio Chatbot(type=messages 기본)에서는 (new_history, state_history) 형태 반환
    return history, history


def launch_ui():
    # ✅ 아이콘을 data-uri로 변환 (서버 요청 없이 브라우저에 바로 표시됨)
    icon_uri = _img_to_data_uri(ICON_FILE)

    with gr.Blocks() as demo:
        # 🔹 헤더 (아이콘 + 타이틀)
        if icon_uri:
            gr.HTML(f"""
            <div style="display:flex; align-items:center; gap:12px; padding:8px 0;">
                <img src="{icon_uri}" style="width:56px; height:56px;" />
                <span style="font-size:1.8rem; font-weight:800;">Agent Pool(30B)</span>
            </div>
            """)
        else:
            # 아이콘 파일이 없을 때도 서비스는 뜨도록
            gr.Markdown("## Agent Pool(30B)")

        # 🔹 여기서는 type 인자 쓰지 않는다. (messages 모드가 기본)
        chatbot = gr.Chatbot(height=480)

        txt = gr.Textbox(label="질문 입력")
        clear = gr.Button("초기화")

        # txt.submit(fn, [입력들], [출력들]) -> fn(message, history)
        txt.submit(
            chat_fn,
            inputs=[txt, chatbot],
            outputs=[chatbot, chatbot]
        )

        # 초기화 버튼: history를 빈 리스트로 리셋
        clear.click(lambda: [], None, chatbot)

    # ✅ file= 서빙이 필요 없으므로 allowed_paths 제거
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)