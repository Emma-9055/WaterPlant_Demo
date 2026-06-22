"""
全局配置：LLM 工厂、路径常量、向量存储工厂入口

LLM 后端（通过 LLM_PROVIDER 或云端 Secrets 切换）：
  1. Groq Cloud（默认）      → 免费 API，跑 Qwen 开源模型，云端部署首选
  2. Ollama 本地              → 零配置本地运行
  3. Anthropic Claude        → 需 ANTHROPIC_API_KEY
  4. OpenAI GPT              → 需 OPENAI_API_KEY

用法：
  # 本地开发（.env 自动加载）
  python app.py

  # Streamlit Cloud（在 Dashboard 设置 Secrets）
  GROQ_API_KEY = "gsk_..."
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- Streamlit Cloud 兼容：从 st.secrets 注入环境变量 ----
def _load_cloud_secrets():
    """在 Streamlit Cloud 环境下，将 Secrets 注入 os.environ"""
    try:
        import streamlit as st
        for key, val in st.secrets.items():
            if key not in os.environ or not os.environ[key]:
                os.environ[key] = str(val)
    except Exception:
        pass  # 非 Streamlit 环境，忽略

_load_cloud_secrets()

# --- 路径 ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# --- LLM 工厂 ---
def get_llm():
    """
    返回 LangChain LLM 实例。
    由 LLM_PROVIDER 环境变量决定后端（默认 groq）。
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "qwen-2.5-32b",  # 开源通义千问，中文能力强
            temperature=0.1,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "qwen3:8b",
            temperature=0.1,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-6",
            temperature=0.1,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            temperature=0.1,
        )

    else:
        raise ValueError(
            f"不支持的 LLM_PROVIDER: {provider}，"
            f"可选: groq, ollama, anthropic, openai"
        )


# --- 向量存储工厂 ---
def get_vector_store():
    """
    返回 VectorStoreBase 实例。
    由 VECTOR_STORE_BACKEND 环境变量控制（默认 faiss）。

    Dify 切换：部署 Dify 后，设 VECTOR_STORE_BACKEND=dify，
    并实现 vector_store/dify_impl.py，Agent 和 UI 代码零改动。
    """
    backend = os.getenv("VECTOR_STORE_BACKEND", "faiss").lower()

    if backend == "faiss":
        from vector_store.faiss_impl import FAISSVectorStore
        return FAISSVectorStore()

    elif backend == "dify":
        raise NotImplementedError(
            "Dify 后端尚未实现。"
            "请创建 vector_store/dify_impl.py 实现 VectorStoreBase 接口，"
            "然后设置 VECTOR_STORE_BACKEND=dify。"
        )

    else:
        raise ValueError(
            f"不支持的 VECTOR_STORE_BACKEND: {backend}，可选: faiss, dify"
        )


if __name__ == "__main__":
    print(f"LLM Provider : {os.getenv('LLM_PROVIDER', 'groq')}")
    print(f"LLM Model    : {os.getenv('LLM_MODEL', 'qwen-2.5-32b (default)')}")
    print(f"Vector Store : {os.getenv('VECTOR_STORE_BACKEND', 'faiss')}")
    print(f"Data Dir     : {DATA_DIR}")
    print(f"FAISS Dir    : {FAISS_INDEX_DIR}")
