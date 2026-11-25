"""
第 5 步（续）：多 Agent 协作 —— 写作-审查流水线
====================================================
多 Agent 的本质不是"很多个模型聊天"，而是**把不同职责封装成图里的不同节点，
每个节点有自己的提示词（角色）和判断标准**。

本例采用生产中最常用的模式（流水线 + 审查回路）：

    用户需求 → [写作 Agent] → 草稿 → [审查 Agent] ─通过→ 结束
                    ↑                        │
                    └──── 不通过 + 修改意见 ←──┘   （带 revision 上限保险丝）

对比单 Agent：写作和审查用不同的 system 提示扮演，且**审查者看不到
写作者的提示词**——职责隔离后，审查更客观，这正是多 Agent 的真正价值。

运行：python3.12 step5b_multi_agent.py   （Mock 模型，无需 API Key）
"""

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ============================================================
# 1. 状态：多 Agent 共享的"工作台"
# 除 messages 外，增加结构化字段：草稿、修改意见、轮次、结论。
# 多 Agent 协作的常见错误是把一切塞进 messages——
# 结构化字段让每个 Agent 的输入输出边界清晰，也更省 token。
# ============================================================

class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    draft: str                 # 当前草稿
    feedback: str              # 审查者的修改意见
    revision: int              # 已修改轮数（保险丝）
    verdict: str               # "approve" | "revise"


# ============================================================
# 2. 两个 Agent：各自独立的角色提示（职责隔离的关键）
# ============================================================

WRITER_PROMPT = ("你是资深文案。根据需求写一条不超过 40 字的产品标语，只输出标语本身。"
                 "若收到修改意见，必须逐条落实。")

REVIEWER_PROMPT = ("你是严格的产品评审。评审标准：①不超过 40 字 ②包含'快'或'稳'字 "
                   "③不使用感叹号。输出 JSON：{\"verdict\": \"approve|revise\", "
                   "\"comment\": \"意见\"}")

class WriterAgent:
    """写作 Agent：封装了自己的提示词，别人看不到"""
    def run(self, requirement: str, feedback: str = "") -> str:
        if feedback:
            return f"稳稳当当地快一步，云服务护航每分钟。"      # 落实意见的第 2 版
        return f"{requirement}，超快超爽，快来用吧！！！"       # 刻意含问题的第 1 版

class ReviewerAgent:
    """审查 Agent：只对草稿负责，按固定标准裁决"""
    def run(self, draft: str) -> dict:
        problems = []
        if len(draft.replace("。", "")) > 40: problems.append("超长")
        if "快" not in draft and "稳" not in draft: problems.append("缺'快'或'稳'字")
        if "！" in draft: problems.append("使用了感叹号")
        if not problems:
            return {"verdict": "approve", "comment": "符合全部标准"}
        return {"verdict": "revise", "comment": "；".join(problems)}

writer, reviewer = WriterAgent(), ReviewerAgent()


# ============================================================
# 3. 节点：每个 Agent 是图中的一个节点
# ============================================================

def writer_node(state: TeamState) -> dict:
    draft = writer.run(state["messages"][0].content, state.get("feedback", ""))
    revision = state.get("revision", 0) + (1 if state.get("feedback") else 0)
    print(f"  [写作Agent 第{revision + 1}稿] {draft}")
    return {"draft": draft, "revision": revision,
            "messages": [AIMessage(content=f"（写作）{draft}")]}

def reviewer_node(state: TeamState) -> dict:
    result = reviewer.run(state["draft"])
    print(f"  [审查Agent] {result['verdict']} —— {result['comment']}")
    return {"verdict": result["verdict"], "feedback": result["comment"],
            "messages": [AIMessage(content=f"（审查）{result['verdict']}:{result['comment']}")]}

def route_after_review(state: TeamState) -> str:
    """条件边：通过 → 结束；不通过 → 打回重写。
    revision >= 2 时强制通过 —— 多 Agent 循环必须有保险丝，否则无限返工"""
    if state["verdict"] == "approve" or state.get("revision", 0) >= 2:
        return END
    return "writer"


# ============================================================
# 4. 组图：writer → reviewer → (approve? END : writer)
# ============================================================

def build():
    g = StateGraph(TeamState)
    g.add_node("writer", writer_node)
    g.add_node("reviewer", reviewer_node)
    g.set_entry_point("writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review)
    return g.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    print("═══ 多 Agent 协作：写作-审查流水线 ═══")
    agent = build()
    final = agent.invoke(
        {"messages": [HumanMessage("给我们的云服务器产品写一条标语")],
         "draft": "", "feedback": "", "revision": 0, "verdict": ""},
        config={"configurable": {"thread_id": "标语任务"}})
    print("\n最终交付:", final["draft"])
    print(f"共经历 {final['revision']} 次修改")
