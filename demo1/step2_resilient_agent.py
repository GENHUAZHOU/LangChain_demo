"""
第 2 步实战：工具容错 + 上下文工程 + 认识 MCP
================================================
在 step1 的循环之上增加三个生产级机制：
  1. 工具失败不崩溃 —— 错误信息作为"观察"回填给模型，让它自己换策略（含重试）
  2. 历史压缩 —— 对话太长时把旧的工具结果折叠成摘要，控制 token 预算
  3. MCP 概念 —— 用一个"Mock MCP 客户端"演示标准化工具接入的形态

运行：python3 step2_resilient_agent.py   （内置 Mock 模型，无需 API Key）
"""

import json
import os
import random
import time
import requests

# ============================================================
# 机制 1：工具容错（生产环境第一课）
# 要点：工具抛异常 ≠ Agent 崩溃。错误文本返回给模型，
# 模型看到错误会自己决定：改参数重试 / 换工具 / 放弃并告知用户。
# ============================================================

class ToolError(Exception):
    pass

def query_api(city: str) -> str:
    """模拟一个不稳定的外部 API：30% 概率超时"""
    if random.random() < 0.3:
        raise ToolError("API 超时（timeout after 3s）")
    return f"{city}：晴，22°C"

def call_tool_with_retry(fn, args, max_retries: int = 2):
    city: str = json.loads(args).get("city")
    """重试包装：指数退避。重试是工程层的责任，模型不懂'等一会再试'。"""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(city)
        except ToolError as e:
            print(f"  [重试 {attempt}/{max_retries}] {e}")
            time.sleep(0.1 * (2 ** attempt))   # 退避：0.2s, 0.4s ...
            err = str(e)
    # 重试耗尽：把失败交给模型，而不是抛异常
    return f"工具失败（已重试 {max_retries} 次）：{err}"

TOOLS = {"query_api": query_api}

# 工具说明书：告诉模型"有哪些工具、参数是什么"。
# 这就是各家 API 的 tools 参数格式（OpenAI 风格，其他家大同小异）。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }
]

# ============================================================
# 机制 2：上下文工程 —— 历史压缩
# 上下文窗口是稀缺资源：工具结果往往又长又多。
# 策略：保留最近的工具结果原文，把更早的折叠成一行摘要。
# （真实框架里对应 LangGraph 的消息裁剪 / 各种 summarization 节点）
# ============================================================

def compact_history(messages, keep_recent: int = 4, token_budget: int = 100):
    """估算字符数当作 token 的粗略代理（中文 ~1.5 字符/token，这里从简）"""
    def size(msgs):
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)

    if size(messages) <= token_budget:
        return messages, False
    # 第一条消息和最后两条信息不参与压缩
    system, rest, last = messages[0], messages[1:len(messages)-2], messages[len(messages)-2:]
    # 找出所有 tool 消息，把"陈旧"的压缩成摘要
    recent_tools = [m for m in rest if m.get("role") == "tool"][::-1][:keep_recent]
    summary_parts, kept = [], []
    for m in rest:
        if m.get("role") == "tool":
            if m in recent_tools:
                kept.append(m)
            else:
                summary_parts.append(f"{m.get('tool_call_id')}: 已执行，结果略")

    summary_msg = {"role": "assistant",
                   "content": f"[历史摘要] 此前工具调用：{'; '.join(summary_parts)}"}
    return [system, summary_msg] + kept + last, True


# ============================================================
# 机制 3：MCP（Model Context Protocol）长什么样
# MCP 把"接入工具"标准化：工具方实现一个 server，Agent 方实现一个 client，
# client 通过 list_tools 发现工具、call_tool 调用 —— 换框架不用重写工具。
# 下面用最简 mock 展示这个接口形态（真实实现走 stdio/HTTP + JSON-RPC）。
# ============================================================

class MockMCPClient:
    """展示 MCP client 的三个核心动作：connect / list_tools / call_tool"""
    def __init__(self, server_name: str):
        self.server_name = server_name

    def list_tools(self):
        # 真实 MCP：向 server 发 tools/list 请求，返回 schema
        return [{"name": "web_search",
                 "description": "搜索网页",
                 "inputSchema": {"type": "object",
                                 "properties": {"query": {"type": "string"}}}}]

    def call_tool(self, name, args):
        # 真实 MCP：发 tools/call 请求
        return {"content": [{"type": "text",
                             "text": f"({self.server_name} 返回) 「{args['query']}」的搜索结果……"}]}



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
        return OpenAIClient()  # 真实模型场景由你按 step1 的 OpenAIClient 接入
    raise EnvironmentError("未设置API KEY")


# ============================================================
# 主循环：step1 循环 + 容错 + 压缩
# ============================================================

def run_agent(question: str, client, max_turns: int = 6) -> str:
    messages = [
        {"role": "system", "content": "你是助手。工具失败时请调整参数重试或换一种问法。"},
        {"role": "user", "content": question},
    ]
    for turn in range(max_turns):
        print(f"\n──── 第 {turn + 1} 轮 ────")
        result = client.chat(messages)
        print(f"[模型] {json.dumps(result, ensure_ascii=False)}")

        if "content" in result:
            return result["content"]

        messages.append({"role": "assistant", "tool_calls": result["tool_calls"]})
        for tc in result["tool_calls"]:
            fn = TOOLS.get(tc["function"]["name"])
            if fn is None:
                output = f"未知工具：{tc["function"]['name']}"      # 错误也回填，让模型知道
            else:
                print(f"[调用] {tc["function"]['name']}{tc["function"]['arguments']}")
                output = call_tool_with_retry(fn, tc["function"]["arguments"])
            print(f"[观察] {output}")
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": str(output)})

        # 每轮结束检查上下文预算，超了就压缩
        messages, compressed = compact_history(messages)
        if compressed:
            print("[上下文] 超出预算，历史已压缩")
    return "达到最大轮数。"


if __name__ == "__main__":
    print("═══ 演示 A1：固定种子，强制第一次调用失败 → 展示重试与模型自适应 ═══")
    random.seed(1)   # 使 query_api 第一次必定抛出超时错误
    print(run_agent("上海天气如何？", make_client()))

    print("\n═══ 演示 A2：随机种子，体验真实的不确定性 ═══")
    random.seed()
    print(run_agent("上海天气如何？", make_client()))

    print("\n═══ 演示 B：MCP 的接口形态 ═══")
    mcp = MockMCPClient("web-search-server")
    print(f"发现的工具: {mcp.list_tools()}")
    print(f"调用结果: {mcp.call_tool('web_search', {'query': 'LangGraph 1.0'})}")
