"""测试：工具安全边界 + create_agent 工具调用循环。

运行::  pytest -v

不需要 API Key，也不联网 —— 用下面的假模型离线跑通完整循环。
"""

from __future__ import annotations

import shutil
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from langchain.tools import ToolException
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent_demo.agent as agent_module
import agent_demo.tools as tools_module
from agent_demo import build_agent
from agent_demo.tools import calculator, get_current_time, read_file, write_file


@pytest.fixture
def sandbox() -> Any:
    """临时的沙箱目录。不用 pytest 的 tmp_path：它依赖系统 TEMP，
    在受限环境里会 PermissionError，测试明明是对的却过不了。"""
    path = Path(__file__).resolve().parent / ".test_tmp"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class ScriptedChatModel(BaseChatModel):
    """按脚本依次吐消息的假模型。

    官方的 ``FakeMessagesListChatModel`` 没实现 ``bind_tools()``，而
    ``create_agent`` 必然调用它（会抛 ``NotImplementedError``），所以自己写。

    用 ``itertools.count()`` 数调用次数，比在 pydantic 模型上改属性省事。
    """

    responses: list[Any] = []
    counter: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.counter is None:
            self.counter = count()
        # 依次吐出预设消息；用完之后一直重复最后一条
        index = min(next(self.counter), len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])


def run_agent(*responses: AIMessage) -> Any:
    """用假模型跑一次 agent，返回完整结果。"""
    agent = build_agent(model=ScriptedChatModel(responses=list(responses)))
    return agent.invoke({"messages": [{"role": "user", "content": "测试"}]})


def calls(result: Any) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------------------
# 安全边界：本项目最该被测试守住的部分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("1+2*3", "7"), ("(128+72)*3/8", "75"), ("2**10", "1024")],
)
def test_计算器正常表达式(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}).endswith(f"= {expected}")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",  # 代码注入
        "open('/etc/passwd').read()",              # 读系统文件
        "x + 1",                                   # 变量
        "1/0",                                     # 除零
        "2**9999",                                 # 指数爆炸
    ],
)
def test_计算器拒绝一切非纯算术(expression: str) -> None:
    """工具参数由模型生成，等于不可信输入 —— 所以不能用 eval。"""
    with pytest.raises(ToolException):
        calculator.invoke({"expression": expression})


@pytest.mark.parametrize("path", ["../outside.txt", "C:/Windows/win.ini", "/etc/passwd"])
def test_文件工具挡住越界路径(path: str) -> None:
    """沙箱：路径解析后必须留在项目目录内。"""
    with pytest.raises(ToolException, match="越界"):
        read_file.invoke({"path": path})


def test_沙箱内可正常读写(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_module, "PROJECT_ROOT", sandbox)
    assert "已写入" in write_file.invoke({"path": "a.txt", "content": "内容"})
    assert "内容" in read_file.invoke({"path": "a.txt"})


def test_时间工具返回星期() -> None:
    assert "星期" in get_current_time.invoke({})


def test_缺密钥时报错可读(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "_load_env", lambda: None)  # 别读真实 .env
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(agent_module.MissingApiKeyError, match="DEEPSEEK_API_KEY"):
        build_agent()


# ---------------------------------------------------------------------------
# 工具调用循环：模型要求调工具 -> 工具真执行 -> 结果回灌 -> 模型给结论
# ---------------------------------------------------------------------------


def test_完整工具调用循环() -> None:
    result = run_agent(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "(128+72)*3/8"}, "id": "c1", "type": "tool_call"},
                {"name": "get_current_time", "args": {}, "id": "c2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="算出来了"),
    )

    executed = calls(result)
    assert [m.name for m in executed] == ["calculator", "get_current_time"]
    assert "75" in str(executed[0].content), "工具返回值应当被回灌"
    assert result["messages"][-1].content == "算出来了"


def test_工具报错回灌给模型而不是崩溃() -> None:
    """LangChain 默认会把工具异常直接抛出去，build_agent 挂了中间件才降级成
    ToolMessage(status="error")，模型因此有机会自我纠正。"""
    result = run_agent(
        AIMessage(
            content="",
            tool_calls=[{"name": "calculator", "args": {"expression": "1/0"}, "id": "c1", "type": "tool_call"}],
        ),
        AIMessage(content="除数为 0，无法计算。"),
    )

    errors = calls(result)
    assert errors and errors[0].status == "error"
    assert "除数为 0" in str(errors[0].content)
