"""
第 6 步实战：生产化三件套 —— 可观测性 / 评测 / 成本控制
========================================================
框架给你循环，生产化靠这三件事。本文件是一个可运行的最小示范：

  ① 可观测性（Tracing）：每次 LLM/工具调用记录 延迟+token+结果 → trace JSONL
     生产替代品：LangSmith / Langfuse / OpenTelemetry，原理完全一样
  ② 评测（Eval）：把"什么算对"写成断言，跑回归测试集
     生产替代品：DeepEval / LangSmith Evaluation
  ③ 成本控制：token 预算记账 + 轮次上限 + 超预算熔断

运行：python3.12 step6_production.py   （Mock 模型，无需 API Key）
"""

import json
import time
from datetime import datetime, timezone

# ============================================================
# ① 可观测性：一个 20 行的 Tracer
# 关键认知：trace 的最小单元是 span（谁、干了什么、多久、花了多少）
# ============================================================

class Tracer:
    def __init__(self, path="trace.jsonl"):
        self.path, self.spans = path, []

    def span(self, name, kind):
        """用上下文管理器包裹一次调用：with tracer.span(...): ..."""
        return _Span(self, name, kind)

    def finish(self, span):
        self.spans.append(span)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    def report(self):
        print("\n─── 观测报告 ───")
        for s in self.spans:
            print(f"  {s['kind']:8s} {s['name']:22s} "
                  f"{s['latency_ms']:6.0f}ms  tokens≈{s['tokens']}  {'OK' if s['ok'] else 'FAIL'}")
        total = sum(s["tokens"] for s in self.spans)
        print(f"  合计 token≈{total}，预算 300，{'未超支' if total <= 300 else '超支！'}")

class _Span:
    def __init__(self, tracer, name, kind):
        self.t, self.data = tracer, {"ts": datetime.now(timezone.utc).isoformat(),
                                     "name": name, "kind": kind, "ok": True,
                                     "tokens": 0, "latency_ms": 0}
    def __enter__(self): self._start = time.perf_counter(); return self
    def __exit__(self, et, ev, tb):
        self.data["latency_ms"] = (time.perf_counter() - self._start) * 1000
        if et: self.data.update(ok=False, error=str(ev))
        self.t.finish(self.data)

# ============================================================
# 被观测的 Agent：沿用 step1 的循环，加上 三件套
# ============================================================

class BudgetExceeded(Exception):
    pass

def llm_call(prompt: str) -> str:
    time.sleep(0.05)                                   # 模拟网络延迟
    return f"关于「{prompt}」的回答"

def tool_call(city: str) -> str:
    time.sleep(0.02)
    return f"{city}：晴，22°C"

def run_agent_with_guardrails(question: str, tracer: Tracer,
                              token_budget: int = 300, max_turns: int = 3) -> str:
    used_tokens = 0
    for turn in range(max_turns):                      # ③ 成本控制之一：轮次上限
        if used_tokens >= token_budget:                # ③ 成本控制之二：预算熔断
            raise BudgetExceeded(f"token 预算 {token_budget} 已耗尽")
        with tracer.span(f"turn{turn+1}_llm", "llm") as s:
            answer = llm_call(question)
            s.data["tokens"] = len(question + answer) // 2   # 粗略估算
        used_tokens += s.data["tokens"]
        if "天气" in question:
            with tracer.span("get_weather", "tool") as s:
                result = tool_call("深圳")
                s.data["tokens"] = len(result) // 2
            used_tokens += s.data["tokens"]
            answer = f"{answer}（{result}）"
            if "详细" in question:
                continue                   # 详细分析类请求：还要第二轮 LLM 汇总
        return answer
    return "达到轮次上限"

# ============================================================
# ② 评测：把"什么算对"写成可执行的断言，形成回归测试集
# 每次改提示词/换模型后重跑，防止质量悄悄劣化
# ============================================================

EVAL_CASES = [
    {"id": "T1", "question": "深圳天气如何？", "must_contain": ["22°C"]},
    {"id": "T2", "question": "什么是 RAG？", "must_contain": ["RAG"]},
    {"id": "T3", "question": "",                "must_not_crash": True},   # 边界用例
]

def run_eval():
    print("═══ 评测报告（回归测试集）═══")
    passed = 0
    for case in EVAL_CASES:
        tracer = Tracer()
        try:
            out = run_agent_with_guardrails(case["question"], tracer)
            ok = all(kw in out for kw in case.get("must_contain", []))
            detail = "通过" if ok else f"失败：输出缺关键词 {case['must_contain']}"
        except Exception as e:
            ok, detail, out = False, f"异常：{type(e).__name__}", ""
        passed += ok
        print(f"  [{case['id']}] {'✅' if ok else '❌'} {detail}")
    print(f"  通过率: {passed}/{len(EVAL_CASES)}")

if __name__ == "__main__":
    import os
    if os.path.exists("trace.jsonl"):
        os.remove("trace.jsonl")

    print("═══ 正常请求：全程被观测 ═══")
    tracer = Tracer()
    print("回答:", run_agent_with_guardrails("深圳天气", tracer))
    tracer.report()

    print("\n═══ 成本熔断演示：详细分析需两轮，预算 10 → 第二轮前熔断 ═══")
    t2 = Tracer()
    try:
        run_agent_with_guardrails("详细分析深圳天气", t2, token_budget=10, max_turns=5)
    except BudgetExceeded as e:
        print("熔断触发:", e)

    run_eval()
    print("\ntrace 已写入 trace.jsonl —— 生产中这就是 LangSmith/Langfuse 的数据源")
