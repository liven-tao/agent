# LangChain 工具调用 Agent

用 **LangChain v1 的 `create_agent`** 构建的工具调用 Agent（Tool-Calling Agent）。
模型能自己决定调用哪个工具、读取工具返回值继续推理，直到给出答案。

**16 个测试覆盖 29 个用例，全部通过**（`pytest`，工具调用循环用假模型离线验证，不需要 API Key）。

**已用真实 DeepSeek 模型端到端验证**，4 个场景全部通过：

| 场景 | 实际表现 |
| --- | --- |
| 多工具并行调用 | 一次发起 `calculator` + `get_current_time`，两个结果都正确 |
| 写文件再读回 | 真的创建了文件，内容一致 |
| 工具报错（1/0） | 不崩溃，错误回灌后模型如实回答并补充数学解释 |
| 越界读 `C:/Windows/win.ini` | 被沙箱拒绝，模型向用户说明原因并给出正确用法 |

## 核心就一行

```python
from langchain.agents import create_agent

agent = create_agent(model=chat_model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
result = agent.invoke({"messages": [{"role": "user", "content": "帮我算 (128+72)*3/8"}]})
```

`create_agent` 返回一个编译好的 LangGraph 图，内部自动完成循环：
**模型 → 发起工具调用 → 执行工具 → 结果回灌 → 模型继续推理**，直到模型不再调用工具。

## 这个项目做了什么

一个 Agent 只有三个部分：**模型 + 工具 + 提示词**。本项目把这三部分拆开，各自处理了几个实际会遇到的问题。

### 1. 工具定义（`agent_demo/tools.py`）

用 `@tool` 装饰器注册 4 个工具。docstring 就是模型看到的工具说明，所以当成「给模型的说明书」来写：

| 工具 | 作用 |
| --- | --- |
| `calculator` | 数学表达式求值 |
| `get_current_time` | 当前日期时间 |
| `read_file` | 读取项目内文件 |
| `write_file` | 写入/追加项目内文件 |

三个工程细节：

**① 计算器用 AST 白名单求值，不用 `eval`。**
`eval("__import__('os').system('rm -rf /')")` 是经典注入漏洞。这里递归遍历表达式树，只放行数字字面量和 `+ - * / // % **` 运算符，函数调用、变量、属性访问一律拒绝。

**② 文件工具带路径沙箱。**
路径 `resolve()` 后校验必须落在项目目录内，挡住 `../outside.txt`、`C:/Windows/win.ini` 这类越界访问。

**③ 工具失败抛 `ToolException`，而不是返回 "error" 字符串。**
异常信息会被当作工具结果回灌给模型，模型能读懂「除数为 0」并换参数重试。

### 2. 工具报错的处理（`agent_demo/agent.py`）

这是踩到坑之后才做对的一步。

**LangChain v1 默认会把工具异常直接抛出去**，整个 Agent 崩掉，模型根本没机会纠正。正确做法是挂 `ToolRetryMiddleware`：

```python
MIDDLEWARE = [ToolRetryMiddleware(max_retries=1, on_failure="continue")]
```

`on_failure="continue"` 会把最终失败包装成 `ToolMessage(status="error")` 回灌给模型：

```
Tool 'calculator' failed after 2 attempts with ToolException: 除数为 0
```

模型拿到这条消息就知道该怎么办。这个行为有测试专门守着（`test_工具报错回灌给模型而不是崩溃`）。

### 3. 模型工厂（写在 `agent.py` 里）

`deepseek-chat` 走 DeepSeek，`gpt-` 开头自动走 OpenAI；缺 Key 时抛可读的 `MissingApiKeyError`，而不是让底层库抛一堆栈。

## 测试策略

`pytest -v test_agent.py` —— **16 个测试 / 29 个用例，全部通过**（171 行）。

关键点：**工具调用循环不需要 API Key，也不联网。**

`create_agent` 一定会对模型调用 `bind_tools()`，而官方的 `FakeMessagesListChatModel` 没有实现它（会抛 `NotImplementedError`）。所以测试里自己写了一个按脚本吐消息的假模型：

```python
class ScriptedChatModel(BaseChatModel):
    def bind_tools(self, tools, **kwargs): return self      # 让 create_agent 能用
    def _generate(self, messages, ...): ...                 # 按顺序返回预设消息
```

于是可以离线断言完整链路：

```python
result = run_agent(ScriptedChatModel(responses=[
    tool_call("calculator", {"expression": "(128+72)*3/8"}),   # 模型要求调工具
    AIMessage(content="(128+72)*3/8 = 75"),                    # 拿到结果后的结论
]))

tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
assert tool_messages[0].name == "calculator"
assert "75" in str(tool_messages[0].content)      # 工具真的执行了
```

测试覆盖三类：

1. **工具行为**：正确性、参数边界，以及两类安全边界（代码注入、路径越界）
2. **组装**：`create_agent` 能起、工具挂上了、缺 Key 时错误可读
3. **循环**：单工具、连续多工具、工具报错回灌

## 运行

```powershell
# 1. 装依赖
pip install -r requirements.txt

# 2. 填密钥
Copy-Item .env.example .env      # 然后填入 DEEPSEEK_API_KEY

# 3. 跑单轮提问
python -m agent_demo "帮我算 (128+72)*3/8，再看下现在几点"
```

在代码里用：

```python
from agent_demo import ask, build_agent

agent = build_agent()
print(ask(agent, "算一下 (128+72)*3/8，再把结果写进 result.txt"))
```

多轮对话（`checkpointer` + 固定 `thread_id`）：

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = build_agent(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "user-1"}}
agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)
agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config)
```

## 加一个自己的工具

只要写个带 docstring 的函数，加进 `TOOLS` 就行：

```python
from langchain.tools import tool

@tool
def word_count(text: str) -> str:
    """统计文本的字符数、词数和行数。"""
    return f"字符 {len(text)}，词 {len(text.split())}，行 {text.count(chr(10)) + 1}"
```

## 文件结构

```
agent_demo/
├─ agent.py     # create_agent 组装 + 系统提示词 + 中间件 + 模型工厂（170 行）
├─ tools.py     # 4 个工具 + 安全校验（AST 白名单、路径沙箱）（175 行）
├─ __init__.py  # 对外 API
└─ __main__.py  # python -m agent_demo 入口
test_agent.py   # 16 个测试 / 29 个用例（171 行）
```

核心代码 **345 行**，且刻意不用 `eval`、字典分发表、`object.__setattr__` 这类
「一眼看不懂」的写法 —— 全部是直白的 `if/elif` 和普通函数。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 必填，模型密钥 |
| `AGENT_MODEL` | 模型名，默认 `deepseek-chat` |
| `AGENT_API_BASE` | 自定义端点（中转 / 本地服务） |
| `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY` | 可选，开启 LangSmith 追踪 |

## 技术栈

Python 3.13 · LangChain 1.2 · LangGraph 1.1 · Pydantic v2 · pytest
