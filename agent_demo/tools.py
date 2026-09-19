"""工具定义：模型的"手"。

`@tool` 里的 docstring 就是模型看到的工具说明，要按"给模型看的说明书"来写。
三个关键点：

1. 参数用 Pydantic 校验 —— 不合法在进函数体前就被拦下
2. 失败抛 `ToolException` —— 错误信息会回灌给模型，模型能自己改参数重试
3. 文件工具带路径沙箱 —— 挡住 `../` 越界，避免模型读到系统文件
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolException, tool
from pydantic import Field

# 文件工具只能访问项目目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAX_READ_CHARS = 20_000


# ---------------------------------------------------------------------------
# 计算器
#
# 为什么不用 eval：工具参数是模型生成的，等于不可信输入。
# eval("__import__('os').system('rm -rf /')") 会直接执行系统命令。
# 所以这里自己遍历表达式的语法树，只放行数字和四则运算。
# ---------------------------------------------------------------------------


def _safe_eval(node: ast.AST) -> float:
    """递归计算表达式的语法树，只允许数字字面量和白名单运算符。"""

    # 一层壳，真正的表达式在 .body 里
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    # 数字字面量：3、2.5 都可以；True/False 是 bool，不算数字
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolException(f"只支持数字字面量，收到了 {node.value!r}")
        return float(node.value)

    # 二元运算：a + b、a ** b 这类
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ToolException("指数过大（>64），拒绝计算")

        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
        except ZeroDivisionError as exc:
            raise ToolException("除数为 0") from exc

        raise ToolException(f"不支持的运算符：{type(node.op).__name__}")

    # 一元运算：-5、+5
    if isinstance(node, ast.UnaryOp):
        value = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ToolException(f"不支持的一元运算符：{type(node.op).__name__}")

    # 函数调用（__import__、open...）和变量一律拒绝
    if isinstance(node, ast.Call):
        raise ToolException("不支持函数调用，只能用 + - * / // % ** 和括号")
    raise ToolException(f"表达式里不支持的语法：{type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式并返回结果。支持 + - * / // % ** 和括号，例如 "(128+72)*3/8"。"""
    if not expression or not expression.strip():
        raise ToolException("expression 不能为空")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolException(f"表达式语法错误：{exc.msg}") from exc

    value = _safe_eval(tree)
    if value != value or value in (float("inf"), float("-inf")):
        raise ToolException("计算结果不是有限数值")
    text = str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else repr(round(value, 10))
    return f"{expression.strip()} = {text}"


# ---------------------------------------------------------------------------
# 时间
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """获取当前日期和时间（含星期）。问"今天几号"时必须调用，不要凭记忆回答。"""
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} 星期{'一二三四五六日'[now.weekday()]}"


# ---------------------------------------------------------------------------
# 文件读写（带沙箱）
# ---------------------------------------------------------------------------


def _safe_path(path: str) -> Path:
    """把路径解析成绝对路径，并确认它没跑出项目目录。"""
    if not path or not path.strip():
        raise ToolException("path 不能为空")

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target = target.resolve()  # resolve() 会把 .. 和符号链接都展开

    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise ToolException(f"路径越界：{path!r} 不在项目目录内，已拒绝访问")
    return target


@tool
def read_file(
    path: Annotated[str, Field(description="相对项目根目录的路径，例如 'README.md'")],
    max_lines: Annotated[int, Field(description="最多读多少行", ge=1, le=500)] = 200,
) -> str:
    """读取项目目录内的文本文件。只能读项目里的文件，越界会被拒绝。"""
    target = _safe_path(path)
    if not target.is_file():
        raise ToolException(f"文件不存在：{path}（只允许项目目录内的文件）")

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    content = "\n".join(lines[:max_lines])[:MAX_READ_CHARS]
    return f"{target.name}（共 {len(lines)} 行，显示前 {min(max_lines, len(lines))} 行）\n{content}"


@tool
def write_file(
    path: Annotated[str, Field(description="相对项目根目录的路径，例如 'notes.txt'")],
    content: Annotated[str, Field(description="要写入的文本内容")],
    mode: Annotated[str, Field(description="overwrite 覆盖，append 追加")] = "overwrite",
) -> str:
    """把文本写入项目目录内的文件。父目录必须已存在，不会自动创建。"""
    target = _safe_path(path)
    if target.is_dir():
        raise ToolException(f"{path} 是目录，不能写入")
    if not target.parent.is_dir():
        raise ToolException(f"父目录不存在：{target.parent}")

    with target.open("a" if mode == "append" else "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"已{'追加' if mode == 'append' else '写入'} {len(content)} 个字符到 {target.name}"


#: 交给 agent 的工具清单
TOOLS = [calculator, get_current_time, read_file, write_file]
