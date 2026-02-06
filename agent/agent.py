# agent/agent.py

from typing import List
from tools.base import Tool


class Agent:
    """
    ReAct 기반 에이전트의 최소 뼈대 
    - Tool 목록을 받아서 보관만 수행
    """

    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}