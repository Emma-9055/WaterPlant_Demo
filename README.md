# 💧 水厂报修智能分类系统

> Water Plant Repair Report Intelligent Classification Agent

基于 **RAG 知识库 + LangChain Agent + Streamlit** 的水厂报修信息智能分类与工单管理系统。

## 🎯 功能

| 功能 | 描述 |
|------|------|
| 🔍 **智能分类** | LLM 自动判断报修类别（管道漏水/水泵故障/电气故障/水质异常/阀门故障） |
| ⚡ **紧急评估** | 自动判定紧急程度（紧急/一般/低） |
| 📚 **RAG 检索** | 在历史案例库中搜索相似故障的处理经验 |
| 📋 **工单管理** | 自动创建工单、指派负责人、跟踪处理状态 |
| 📞 **联系方式** | 查询各水厂的联系人信息 |

## 🏗️ 架构

```
Streamlit 卡片 UI
      │
      ▼
LangChain Agent (ReAct 推理循环)
      │
      ├── classify_repair_report  ──→ LLM 分类
      ├── search_similar_cases    ──→ FAISS 向量检索
      ├── create_work_order       ──→ 创建工单
      ├── get_plant_contact       ──→ 查询联系方式
      └── update_work_order_status ──→ 更新状态
      │
      ▼
FAISS 知识库（32 条记录：20 案例 + 12 维修指南）
```

## 🚀 快速开始

### 前置条件

```bash
# 安装 Ollama 并拉取模型（开源免费，无需 API Key）
ollama pull qwen3:8b

# 或使用 Anthropic Claude（效果更好）
# 编辑 .env，设置 LLM_PROVIDER=anthropic 并填入 ANTHROPIC_API_KEY
```

### 安装运行

```bash
pip install -r requirements.txt
python app.py
# 浏览器打开 http://localhost:8501
```

## 📂 项目结构

```
├── app.py                      # 入口：初始化 + 启动
├── config.py                   # LLM 工厂 + 向量库工厂
├── models.py                   # Pydantic 数据模型
├── data/
│   └── seed_data.py            # Mock 数据（5 水厂 + 20 案例 + 12 文档）
├── vector_store/
│   ├── __init__.py             # 抽象基类 + 工厂（Dify 切换点）
│   └── faiss_impl.py           # FAISS 本地实现
├── agent/
│   ├── tools.py                # 5 个 LangChain 工具
│   └── orchestrator.py         # ReAct Agent + System Prompt
└── ui/
    └── app.py                  # Streamlit 卡片 UI
```

## 🔧 定制指南

| 你想改什么 | 改哪个文件 |
|-----------|-----------|
| 知识库内容（水厂/案例/维修指南） | `data/seed_data.py` |
| Agent 的行为逻辑 | `agent/orchestrator.py` 的 `SYSTEM_PROMPT` |
| 分类标准和紧急度判定规则 | `agent/tools.py` 的 `_classify_with_llm()` |
| 前端样式和卡片布局 | `ui/app.py` |
| 切换 LLM（Ollama/Claude/GPT） | `.env` 文件 |

## 🔌 Dify 切换

当 Dify 就绪后，只需：
1. 新建 `vector_store/dify_impl.py` 实现 `VectorStoreBase` 接口
2. `.env` 中设 `VECTOR_STORE_BACKEND=dify`
3. Agent 和 UI 代码**零改动**

## 📄 License

MIT
