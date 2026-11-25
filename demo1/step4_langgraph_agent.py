"""
第 4 步实战：用 LangGraph 重写手写 Agent
============================================
step1 我们手写了 while 循环管理状态；step4 交给框架。对照关系：

    手写版                          LangGraph 版
    ─────────────────────────       ─────────────────────────
    messages 列表（全局变量）  →    State（TypedDict + reducer）
    for 循环                   →    图的边自动流转（agent → tools → agent）
    if 有 tool_calls: 执行     →    条件边（conditional edge）
    messages.append(...)       →    节点返回增量，reducer 自动合并
    （没有）                   →    checkpoint：每步自动存盘，可中断/恢复/回放

运行：python3 step4_langgraph_agent.py   （内置 Mock 模型，无需 API Key）
"""
import os
from typing import Annotated, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 定义状态：Agent 的全部记忆就是 state["messages"]
# Annotated[..., add_messages] 是关键：add_messages 是 reducer，
# 节点只返回"增量"消息，框架自动 append 到已有列表 —— 取代手写 messages.append
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 2. 工具（与 step1 完全相同的两个工具）
# ============================================================

def get_weather(city: str) -> str:
    return {"北京": "晴，12°C", "上海": "小雨，18°C", "深圳": "多云，26°C"}[city]

def calculator(expression: str) -> str:
    return str(round(eval(expression), 1))

TOOLS = {"get_weather": get_weather, "calculator": calculator}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_weather",
        "description": "查询城市天气",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator",
        "description": "计算数学表达式",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

def make_llm():
    """有 OPENAI_API_KEY 时可换成真实 ChatOpenAI，图的结构完全不用改"""
    """创建模型客户端。OPENAI_API_KEY / OPENAI_BASE_URL 由 SDK 自动读取。"""
    params = {"model": os.environ.get("OPENAI_MODEL"), "api_key": os.environ["OPENAI_API_KEY"],
              "base_url": os.environ.get("OPENAI_BASE_URL")}
    return ChatOpenAI(**params).bind_tools(TOOLS_SCHEMA)
    # 真实模型写法：
    #   from langchain_openai import ChatOpenAI
    #   return ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS_SCHEMA)


# ============================================================
# 4. 定义节点：每个节点 = 接收 state，干一件事，返回增量
# ============================================================

llm = make_llm()

def agent_node(state: AgentState) -> dict:
    """『思考』节点：调 LLM。对应手写版循环里的 client.chat(messages)"""
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}

def tool_node(state: AgentState) -> dict:
    """『行动』节点：执行 LLM 请求的所有工具。对应手写版的 for tc in tool_calls"""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        output = TOOLS[tc["name"]](**tc["args"])
        print(f"    [执行工具] {tc['name']}{tc['args']} → {output}")
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    """条件边：模型还想调工具 → 去 tools 节点；否则 → 结束。对应手写版的 if 判断"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ============================================================
# 5. 组装图：节点 + 边。这就是 LangGraph 的核心心智模型
# ============================================================

def build_agent(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)      # 思考节点
    graph.add_node("tools", tool_node)       # 行动节点

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue,
                                {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")         # 行动完回到思考 —— 循环就这一条边

    # compile 时挂上 checkpointer → 每一步自动 checkpoint
    return graph.compile(checkpointer=checkpointer)


# ============================================================
# 6. 运行：多轮对话 + 状态持久化演示
# ============================================================

if __name__ == "__main__":
    memory = MemorySaver()
    agent = build_agent(checkpointer=memory)
    config = {"configurable": {"thread_id": "会话1"}}   # 同一 thread = 同一份持久状态

    print("═══ 第 1 轮对话（会话1）═══")
    r1 = agent.invoke({"messages": [HumanMessage("深圳天气如何？换算成华氏度？")]}, config)
    print("最终回复:", r1["messages"][-1].content)

    print("\n═══ 第 2 轮对话：不重复上文，框架靠 checkpoint 记得之前的消息 ═══")
    r2 = agent.invoke({"messages": [HumanMessage("谢谢！你都会什么？")]}, config)
    print("最终回复:", r2["messages"][-1].content)

    print("\n═══ checkpoint 里存了什么？ ═══")
    state = agent.get_state(config)
    print(f"状态中消息条数: {len(state.values['messages'])}")
    print("消息序列:")
    for m in state.values["messages"]:
        role = type(m).__name__.replace("Message", "")
        brief = (m.content or f"tool_calls={[t['name'] for t in m.tool_calls]}")[:40]
        print(f"  [{role:9s}] {brief}")

    print("\n═══ 另一个 thread 是独立的（隔离会话）═══")
    r3 = agent.invoke({"messages": [HumanMessage("你好")]},
                      {"configurable": {"thread_id": "会话2"}})
    print("新会话回复:", r3["messages"][-1].content)
    print("新会话消息条数:", len(agent.get_state({"configurable": {"thread_id": "会话2"}})
                                .values["messages"]))
