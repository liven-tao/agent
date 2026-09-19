"""基于 LangChain v1 ``create_agent`` 的工具调用 Agent。

:: 

    from agent_demo import ask, build_agent

    agent = build_agent()
    print(ask(agent, "帮我算 (128+72)*3/8"))
"""

from .agent import MissingApiKeyError, ask, build_agent
from .tools import TOOLS

__version__ = "1.0.0"

__all__ = ["MissingApiKeyError", "TOOLS", "__version__", "ask", "build_agent"]
