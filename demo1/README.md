
# Agent技术栈

## 1. 协议层

- **MCP**：Agent ↔ 工具/数据源的标准接口（2026 年移交 Linux 基金会，事实标准）。
- **A2A**：跨框架 Agent 之间发现与协作的标准协议（CrewAI、Google ADK 原生支持）。

## 2. 记忆与上下文工程

- 短期记忆：会话内上下文管理（各框架基本具备）。
- 长期记忆：**普遍短板**——生产级长程记忆、语义检索、自动整理尚无框架完善解决，常借助向量库（Pinecone、Qdrant、pgvector）自建。
- 上下文工程（预算/缓存/压缩）是开发者自己的责任。

## 3. RAG 与检索

- LlamaIndex、Haystack 专长；LangChain 拥有最大集成目录。
- 典型组合：文档解析 → 向量/混合检索 → 重排 → Agent 工具调用。

## 4. 可观测性与评测

- **Tracing**：LangSmith、Langfuse（开源）、Arize Phoenix、OpenAI 内置 tracing。
- **评测**：DeepEval、LangSmith Evaluation；注意多数 benchmark 只测任务完成率，安全维度缺失。
- 实践：对工具函数做单元测试，用 mock LLM 做集成测试，上线前压测。

## 5. 生产工程要素

- **持久化执行**：checkpoint（LangGraph + Postgres/Redis）、必要时引入 Temporal。
- **成本控制**：设置 max\_turns/迭代上限、token 预算、多模型路由（LangGraph 比 CrewAI 省 15–18% token）。
- **错误处理**：重试退避、fallback 工具、人工升级路径。
- **HITL**：人工审批门不是事后补丁，应从设计之初纳入。

