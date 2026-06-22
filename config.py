"""
全局配置：LLM 工厂、路径常量、向量存储工厂入口

LLM 优先级（开箱即用，无需 API Key）：
  1. Ollama 本地模型（默认）  → 零配置，开源友好
  2. Anthropic Claude       → 需 ANTHROPIC_API_KEY
  3. OpenAI GPT             → 需 OPENAI_API_KEY

用法：
  # 默认使用 Ollama（需先 ollama pull qwen3）
  python app.py

  # 使用 Anthropic
  set LLM_PROVIDER=anthropic && python app.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- 路径 ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# --- LLM 工厂 ---
def get_llm():
    """
    返回 LangChain LLM 实例。
    由 LLM_PROVIDER 环境变量决定后端（默认 ollama）。
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL", "")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "qwen3:8b",   # 开源友好，中文能力强
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
            f"可选: ollama, anthropic, openai"
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
    print(f"LLM Provider : {os.getenv('LLM_PROVIDER', 'ollama')}")
    print(f"LLM Model    : {os.getenv('LLM_MODEL', 'qwen3:8b (default)')}")
    print(f"Vector Store : {os.getenv('VECTOR_STORE_BACKEND', 'faiss')}")
    print(f"Data Dir     : {DATA_DIR}")
    print(f"FAISS Dir    : {FAISS_INDEX_DIR}")
