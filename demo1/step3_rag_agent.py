"""
第 3 步实战：RAG —— 让 Agent 回答"文档里写了什么"
====================================================
RAG 流水线（本文件每个函数对应一环）：

    离线：文档 → ① 切分(chunk) → ② 向量化(embed) → 存入"向量库"
    在线：用户问题 → ② 向量化 → ③ 相似度检索(retrieve) → ④ 注入上下文 → LLM 生成

不用任何外部服务：向量用"字符 3-gram 词袋"表示，相似度用余弦。
它检索精度不如真实 embedding 模型，但机制完全一样，
换成 OpenAI/BGE embedding 只是换 embed() 函数的内部实现。

运行：python3 step3_rag_agent.py   （Mock 模型，无需 API Key）
"""

import math
import os
import re
from collections import Counter

# ============================================================
# ① 切分（Chunking）
# 为什么切分？上下文窗口有限 + 检索要"准"：整篇文档太长，
# 检索到的是"相关段落"而不是整篇。chunk 要有重叠，防止答案被切断。
# ============================================================

def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list:
    """按字符滑窗切分（生产中常按句/段落 + 语义切分，原理相同）"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap          # 重叠 30 字符：边界处的句子不会丢
    return [c.strip() for c in chunks if c.strip()]


# ============================================================
# ② 向量化（Embedding）
# 文本 → 数字向量。相似文本的向量应该"方向接近"。
# 这里用字符 3-gram 词袋：把文本拆成所有 3 字符片段，统计出现次数。
# 真实模型（BGE/text-embedding-3）做的是同一件事，只是向量更"懂语义"。
# ============================================================

def embed(text: str) -> dict:
    text = re.sub(r"\s+", "", text)
    grams = Counter(text[i:i+3] for i in range(len(text) - 2))
    # L2 归一化：使余弦相似度只看方向，不看长度
    norm = math.sqrt(sum(v * v for v in grams.values())) or 1
    return {k: v / norm for k, v in grams.items()}


# ============================================================
# ③ 检索（Retrieve）
# 余弦相似度 = 两向量点积（已归一化）。取 Top-K 最相似的 chunk。
# ============================================================

def cosine(a: dict, b: dict) -> float:
    return sum(v * b.get(k, 0) for k, v in a.items())

class VectorStore:
    """最简向量库：内存字典。生产中是 pgvector/Qdrant（多的是 ANN 索引）"""
    def __init__(self):
        self.records = []          # [(vector, chunk_text, meta)]

    def add(self, chunks: list, meta: dict):
        for i, c in enumerate(chunks):
            self.records.append((embed(c), c, {**meta, "chunk_id": i}))

    def search(self, query: str, top_k: int = 2) -> list:
        qv = embed(query)
        scored = [(cosine(qv, v), text, meta) for v, text, meta in self.records]
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


# ============================================================
# 准备知识库：一篇关于"公司差旅报销制度"的文档（LLM 训练数据里没有）
# 这就是 RAG 的意义：让通用模型回答"私域知识"。
# ============================================================

COMPANY_DOC = """
公司差旅报销制度（2026修订版）。第一条：交通费报销标准。员工乘坐高铁一律报销二等座，
仅总监及以上职级可报销一等座。市内交通费每人每日上限80元，需提供行程票据。
第二条：住宿费报销标准。一线城市每晚上限500元，二线城市每晚上限350元，
超标部分自行承担，因会议酒店统一安排产生的超标可申请豁免。
第三条：报销流程与时效。差旅结束后15个工作日内提交报销单，逾期系统自动关闭，
特殊情况需部门总监邮件审批延期。所有票据须为增值税发票，定额发票自2026年起不再受理。
第四条：审批权限。单笔报销2000元以下由直属经理审批，2000至10000元由总监审批，
超过10000元需财务总监加签。报销款项于审批通过后5个工作日内到账。
""".replace("\n", "").strip()

store = VectorStore()
store.add(chunk_text(COMPANY_DOC), meta={"source": "差旅报销制度2026"})

# ============================================================
# ④ 注入上下文 + 生成
# 关键模式：把检索到的片段放进 system 提示，并附上出处（可溯源）。
# ============================================================

def build_rag_messages(question: str, hits: list) -> list:
    context = "\n".join(f"[片段{i+1}] {text}" for i, (score, text, meta) in enumerate(hits))
    return [
        {"role": "system",
         "content": f"你是公司制度助手。请仅依据以下检索到的制度片段回答，"
                    f"并注明引用了哪个片段；片段中没有的信息要回答'制度中未提及'。\n{context}"},
        {"role": "user", "content": question},
    ]


# ============================================================
# Mock 模型：演示"忠实于检索结果"的回答行为
# ============================================================

class MockLLM:
    def chat(self, messages):
        system = messages[0]["content"]
        question = messages[-1]["content"]
        # 从 system 上下文里"找答案"——模拟 LLM 依据片段作答
        if "一等座" in question:
            snippet = [s for s in system.split("[片段") if "一等座" in s]
            return {"content": f"制度规定高铁一律报销二等座，仅总监及以上职级可报销一等座。"
                               f"（依据{snippet[0][0:2].strip('] ')}）" if snippet else
                               "制度中未提及。"}
        if "住宿费报销标准" in question:
            snippet = [s for s in system.split("[片段") if "住宿费" in s]
            return {"content": "制度规定：一线城市每晚 500 元，二线城市每晚 350 元，"
                               "超标部分自行承担。（依据制度片段）"}
        if "25" in question:   # 文档里没有"25个工作日"，考察能否抑制幻觉
            return {"content": "制度中未提及 25 个工作日的时效规定。检索片段仅说明："
                               "差旅结束后 15 个工作日内提交，逾期系统自动关闭。（依据片段）"}
        return {"content": f"已根据检索到的制度片段回答：「{question}」相关内容见引用片段。"}

def make_client():
    if os.environ.get("OPENAI_API_KEY"):
        return MockLLM()   # 有 Key 时按 step1 的 OpenAIClient 接入真实模型
    return MockLLM()

def answer(question: str, client, top_k: int = 2) -> str:
    # 检索
    hits = store.search(question, top_k=top_k)
    print(f"\n[问题] {question}")
    for i, (score, text, meta) in enumerate(hits):
        print(f"  [检索片段{i+1}] 相似度={score:.3f} | {text[:40]}...")
    # 注入 + 生成
    messages = build_rag_messages(question, hits)
    return client.chat(messages)["content"]


if __name__ == "__main__":
    print("═══ 知识库 ═══")
    chunks = chunk_text(COMPANY_DOC)
    print(f"文档 {len(COMPANY_DOC)} 字符 → 切分为 {len(chunks)} 个 chunk（含重叠）")

    print("\n═══ 场景 1：答案在文档里 ═══")
    print("[回答]", answer("出差坐高铁能报销一等座吗？", make_client()))

    print("\n═══ 场景 2：换个问法仍能命中（词面一致时） ═══")
    print("[回答]", answer("住宿费报销标准是多少？", make_client()))

    print("\n═══ 场景 2b：【重要教训】同义改写会检索失败 ═══")
    print("[检索演示]", answer("住酒店每晚最多报多少钱？", make_client()))
    print(">>> 注意：「住酒店」与文档中的「住宿费」没有任何共同字符片段，")
    print(">>> 相似度为 0 —— 这就是词法检索(n-gram/BM25)的弱点。")
    print(">>> 真实 embedding 模型把语义相近的文本映射到相近向量，")
    print(">>> 这类同义改写也能命中。生产中常用 混合检索(词法+向量) + 重排。")

    print("\n═══ 场景 3：答案不在文档里 —— 抑制幻觉 ═══")
    print("[回答]", answer("报销时限是25个工作日吗？", make_client()))
