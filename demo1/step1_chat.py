"""
最小 Agent 循环 —— 第 1 步实战
=================================
Agent = LLM + 工具 + 循环 + 状态。这个文件就是这四个部分的全部。

运行方式：
    python3 step1_chat.py
    OPENAI_API_KEY=sk-xxx python3 step1_chat.py
"""

import json
import os
import re
import requests

# ============================================================
# 第一部分：工具（Tools）
# 工具就是普通 Python 函数。关键是给每个函数配一份"说明书"（schema），
# LLM 靠说明书决定什么时候调用、传什么参数。
# ============================================================

def get_weather(city: str) -> str:
    """查询指定城市的天气（演示用假数据）"""
    fake_db = {"北京": "晴，12°C", "上海": "小雨，18°C", "深圳": "多云，39°C"}
    return fake_db.get(city, f"{city}：暂无数据")

def calculator(expression: str) -> str:
    """计算一个数学表达式，如 12*34-5"""
    try:
        # 生产环境不要用 eval！这里仅演示。真实项目用 ast 安全求值。
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expression):
            return "错误：表达式包含非法字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算失败：{e}"

TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
}

# 工具说明书：告诉模型"有哪些工具、参数是什么"。
# 这就是各家 API 的 tools 参数格式（OpenAI 风格，其他家大同小异）。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 12*34-5"}},
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# 第二部分：LLM 封装
# 两种实现：真实 API（OpenAI 兼容格式）和 Mock（离线演示用）。
# 两者的接口完全一样：输入消息列表，返回 {回复 或 工具调用}。
# ============================================================

class OpenAIClient:
    """真实模型：走 OpenAI 兼容的 chat/completions 接口"""
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, messages):
        resp = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "tools": TOOLS_SCHEMA},
            timeout=60,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        if msg.get("tool_calls"):
            return {"tool_calls": [
                {"id": tc["id"], 'type': 'function', "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                }
                for tc in msg["tool_calls"]
            ]}
        return {"content": msg.get("content", "")}

def make_client():
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    raise EnvironmentError("未设置API KEY")


# ============================================================
# 第三部分：循环 + 状态（Agent 的心脏）
# messages 列表就是"状态"。每一轮：模型思考 → 执行工具 → 结果回填 → 再思考。
# ============================================================

def run_agent(user_question: str, client, max_turns: int = 5) -> str:
    # 状态 1：系统提示（人设与规则）+ 用户问题
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手，可以调用工具。回答要简洁。"},
        {"role": "user", "content": user_question},
    ]

    for turn in range(max_turns):          # max_turns 防止失控循环烧钱
        print(f"\n──── 第 {turn + 1} 轮 ────")

        # ① 思考：把全部历史交给模型
        print(f"[请求信息]{messages}")
        result = client.chat(messages)
        print(f"[模型输出] {json.dumps(result, ensure_ascii=False)}")

        # ② 如果模型直接给出答案，循环结束
        if "content" in result:
            return result["content"]

        # ③ 模型发起工具调用 → 我们（不是模型！）真正执行函数
        # 注意：模型只会"说"要调什么工具，执行权永远在你手里 —— 这就是安全边界
        messages.append({"role": "assistant",
                         "content": "",
                         "tool_calls": result["tool_calls"]})
        for tc in result["tool_calls"]:
            fn = TOOLS[tc["function"]["name"]]
            print(f"[执行工具] {tc["function"]['name']}({tc["function"]['arguments']})")
            output = fn(**json.loads(tc["function"]["arguments"]))
            print(f"[工具结果] {output}")
            # ④ 观察：结果作为 tool 消息回填进状态，供下一轮思考
            messages.append({"role": "tool",
                             "tool_call_id": tc.get("id", f"t{turn}"),
                             "content": str(output)})

    return "达到最大轮数上限，已停止。"


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    answer = run_agent(
        "深圳现在天气怎么样？这个温度换算成华氏度是多少？",
        make_client(),
    )
    print("\n════ 最终答案 ════")
    print(answer)
