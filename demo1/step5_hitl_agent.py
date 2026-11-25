"""
第 5 步实战：人工审批（Human-in-the-loop）—— LangGraph interrupt() 
=====================================================================
场景：报销提交流程。≤2000 元自动通过；>2000 元必须暂停，
等人工批准或拒绝后才能继续。这就是"Agent 有执行权但敏感操作有刹车"。

核心机制（建立在 step4 checkpoint 之上）：
    interrupt(payload)  →  图在节点内暂停，payload 交给人类，
                           整个状态已自动 checkpoint 存盘
    Command(resume=X)   →  人类把决定传回来，节点从头重跑，
                           这次 interrupt() 直接返回 X，图继续

运行：python3.12 step5_hitl_agent.py   （Mock 模型，无需 API Key）
"""

import json
import logging
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI

# ============================================================
# 状态与工具
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def auto_check(amount: float) -> str:
    """自动校验：低于 2000 无需人工"""
    return f"金额 {amount} 元 ≤ 2000 元，自动校验通过。"

def notice(amount: float, reason: str) -> str:
    """敏感操作：提交报销单（模拟调公司财务系统）"""
    return f"通知成功：{reason}，金额 {amount} 元。"

def submit_reimbursement(amount: float, reason: str) -> str:
    # ★ 关键：暂停！整个状态已 checkpoint。
    # payload 会展示给人类；节点从此处"冻结"。
    decision = interrupt({
        "question": "报销金额超过 2000 元，需要人工审批",
        "tool": "submit_reimbursement",  # ★ 必须是字符串！函数对象无法 msgpack 序列化进 checkpoint
        "args": amount,
    })
    # 恢复后节点从头重跑，interrupt() 这次直接返回人类的决定
    if decision.get("approved"):
        return f"报销单已提交：{reason}，金额 {amount} 元。单号 TX-{int(amount)}026"
    else:
        return "提交已被人工拒绝"


SAFE_TOOLS = {"auto_check": auto_check, "submit_reimbursement": submit_reimbursement, "notice": notice}

THRESHOLD = 2000.0

# 注意：JSON Schema 没有 "float" 类型，金额必须用 "number"，
# 否则部分 MaaS 网关会在入口组件做严格校验直接返回 400
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "auto_check",
        "description": "报销金额 <= 2000元",
        "parameters": {"type": "object",
            "properties": {"amount": {"type": "number", "description": "报销金额，单位元"}},
            "required": ["amount"]}}},
    {"type": "function", "function": {"name": "submit_reimbursement",
        "description": "报销金额 > 2000元",
        "parameters": {"type": "object",
            "properties": {"amount": {"type": "number", "description": "报销金额，单位元"},
                           "reason": {"type": "string", "description": "报销事由"}},
            "required": ["amount", "reason"]}}},
    {"type": "function", "function": {"name": "notice",
        "description": "报销成功或失败通知报销人",
        "parameters": {"type": "object",
            "properties": {"amount": {"type": "number", "description": "报销金额，单位元"},
                           "reason": {"type": "string", "description": "通知事由"}},
            "required": ["amount", "reason"]}}},
]

# ============================================================
# 节点：思考 + 行动（含 interrupt 的敏感工具处理）
# ============================================================
def make_llm():
    """有 OPENAI_API_KEY 时可换成真实 ChatOpenAI，图的结构完全不用改"""
    """创建模型客户端。OPENAI_API_KEY / OPENAI_BASE_URL 由 SDK 自动读取。"""
    params = {"model": os.environ.get("OPENAI_MODEL"), "api_key": os.environ["OPENAI_API_KEY"],
              "base_url": os.environ.get("OPENAI_BASE_URL")}
    return ChatOpenAI(**params).bind_tools(TOOLS_SCHEMA)

llm = make_llm()

def agent_node(state: AgentState) -> dict:
    res = llm.invoke(state["messages"])
    return {"messages": [res]}

def tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        if name in SAFE_TOOLS:
            output = SAFE_TOOLS[name](**args)
        else:
            output = f"未知工具 {name}"
        results.append(ToolMessage(content=output, tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


# ============================================================
# 组装与运行
# ============================================================

def build():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())

def show_state(agent, config, label):
    snap = agent.get_state(config)
    if snap.next:
        # 下一步要去哪个节点？interrupt 的 payload 是什么？
        task = snap.tasks[0]
        payload = getattr(task, "interrupts", None)
        info = payload[0].value if payload else None
        print(f"[{label}] 已暂停，下一步待执行: {snap.next}，等待输入: {json.dumps(info, ensure_ascii=False)}")
    else:
        print(f"[{label}] 流程已完成")

if __name__ == "__main__":
    agent = build()

    # ── 场景 A：800 元 → 不触发审批，全自动 ──
    print("═══ 场景 A：800 元报销（≤2000，无需人工）═══")
    cfg_a = {"configurable": {"thread_id": "A"}}
    r = agent.invoke({"messages": [HumanMessage("提交一笔 800 元差旅报销")]}, cfg_a)
    print("最终回复:", r["messages"][-1].content, "\n")

    # ── 场景 B：5000 元 → 暂停，人工【批准】 ──
    print("═══ 场景 B：5000 元报销（>2000，人工批准）═══")
    cfg_b = {"configurable": {"thread_id": "B"}}
    agent.invoke({"messages": [HumanMessage("提交一笔 5000 元差旅报销")]}, cfg_b)
    show_state(agent, cfg_b, "暂停时")          # 图冻结在这里，可关机、可换进程

    r = agent.invoke(Command(resume={"approved": True}), cfg_b)   # 人类：批准
    print("恢复后最终回复:", r["messages"][-1].content, "\n")

    # ── 场景 C：5000 元 → 暂停，人工【拒绝】 ──
    print("═══ 场景 C：5000 元报销（>2000，人工拒绝）═══")
    cfg_c = {"configurable": {"thread_id": "C"}}
    agent.invoke({"messages": [HumanMessage("提交一笔 9000 元差旅报销")]}, cfg_c)
    show_state(agent, cfg_c, "暂停时")
    r = agent.invoke(Command(resume={"approved": False}), cfg_c)  # 人类：拒绝
    print("恢复后最终回复:", r["messages"][-1].content)
