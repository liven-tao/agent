"""用 LangChain v1 的 ``create_agent`` 组装 agent。

核心就三个参数：

    agent = create_agent(model=..., tools=..., system_prompt=...)

它返回一个编译好的 LangGraph 图，内部自动完成循环：
模型 -> 发起工具调用 -> 执行工具 -> 结果回灌 -> 模型继续推理，直到给出答案。

直接运行本文件可看演示::

    python -m agent_demo "帮我算 (128+72)*3/8"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .tools import TOOLS

DEFAULT_MODEL = "deepseek-chat"

# 第 1 条最关键：不明确要求"必须调用工具"，模型容易凭记忆瞎算
SYSTEM_PROMPT = """你是一个可以调用工具的助手，请遵守：

1. 需要计算、查时间、读写文件时，必须先调用对应工具，不要凭记忆或心算。
2. 可以一次发起多个工具调用；拿到结果后若还需再算一步，就继续调用。
3. 工具报错时先读懂错误信息，修正参数后重试；仍失败就如实说明。
4. 文件工具只能访问项目目录内的路径，越界会被拒绝，不要反复尝试。
5. 用户没有指定语言时用中文回答；结果要简洁，数值带上说明。
"""

# LangChain 默认会把工具异常直接抛出去、整个 agent 崩掉。
# on_failure="continue" 会把最终失败包装成 ToolMessage(status="error") 回灌给模型，
# 模型才有机会自己改参数重试。
MIDDLEWARE = [ToolRetryMiddleware(max_retries=1, on_failure="continue")]


class MissingApiKeyError(RuntimeError):
    """没有找到 API Key。"""


def _load_env() -> None:
    """加载项目根目录的 .env（可选依赖 python-dotenv）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    path = Path(__file__).resolve().parent.parent / ".env"
    if path.is_file():
        load_dotenv(path)


def _chat_model(model: str | None, temperature: float) -> BaseChatModel:
    """按模型名造出对应厂商的模型实例。``gpt-`` 开头自动走 OpenAI。"""
    _load_env()

    name = model or os.getenv("AGENT_MODEL") or DEFAULT_MODEL
    key_env = "OPENAI_API_KEY" if name.startswith("gpt-") else "DEEPSEEK_API_KEY"
    api_key = os.getenv(key_env)
    if not api_key:
        raise MissingApiKeyError(
            f"缺少 {key_env}。请复制 .env.example 为 .env 并填入密钥，"
            "或设置环境变量后重试。"
        )

    base_url = os.getenv("AGENT_API_BASE")
    if name.startswith("gpt-"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=temperature, api_key=api_key, base_url=base_url)

    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url or "https://api.deepseek.com",
    )


def build_agent(
    model: str | BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
    checkpointer: Any | None = None,
) -> Any:
    """创建 agent。

    Args:
        model: 模型名（默认 ``deepseek-chat``），或直接传一个已经构造好的
            ``BaseChatModel`` 实例（单元测试里很有用，可以喂假模型）。
        temperature: 采样温度，默认 0（工具调用场景更稳定）。
        checkpointer: 传 ``InMemorySaver()`` 即可支持多轮记忆（配合固定的
            ``thread_id`` 使用）。

    Returns:
        编译好的 agent，用 ``agent.invoke({"messages": [...]})`` 调用。

    Example:
        >>> agent = build_agent()
        >>> result = agent.invoke({"messages": [{"role": "user", "content": "1+1"}]})
        >>> result["messages"][-1].content
    """
    # model 既可以是模型名，也可以直接是构造好的模型对象（测试时喂假模型用）
    if isinstance(model, BaseChatModel):
        chat_model = model
    else:
        chat_model = _chat_model(model, temperature)

    return create_agent(
        model=chat_model,
        tools=list(TOOLS),
        system_prompt=SYSTEM_PROMPT,
        middleware=list(MIDDLEWARE),
        checkpointer=checkpointer,
        name="tool_agent",
    )


def ask(agent: Any, question: str) -> str:
    """问一句、拿最终回复文本。

    多轮对话用 ``InMemorySaver`` + 固定 ``thread_id``，见 README。
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return message_text(result["messages"][-1])


def message_text(message: BaseMessage | Any) -> str:
    """把回复归一化成字符串。

    部分模型（如 deepseek-reasoner）返回的是 content block 列表，
    直接打印会很难看，这里只取其中的文本块。
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def main() -> int:
    """命令行演示：``python -m agent_demo "你的问题"``。"""
    question = " ".join(sys.argv[1:]).strip() or "帮我算一下 (128+72)*3/8，再看下现在几点"
    try:
        agent = build_agent()
    except MissingApiKeyError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"问：{question}\n")
    print(ask(agent, question))
    return 0
