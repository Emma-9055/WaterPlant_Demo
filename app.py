"""
水厂报修智能分类系统 —— 本地开发入口

本地运行:
    python app.py

云端部署:
    Streamlit Cloud 入口为 ui/app.py
    在 Dashboard 设置 Secrets:
      GROQ_API_KEY = "gsk_..."
"""
import sys
import os
from pathlib import Path


def check_env():
    """检查环境配置（本地 .env 或云端 Secrets）"""
    from dotenv import load_dotenv
    load_dotenv()

    # 检查 Groq API Key
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "gsk_...":
        return True

    # 检查 Anthropic 作为备选
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and anthropic_key != "sk-ant-...":
        return True

    # 检查 Ollama
    if os.getenv("LLM_PROVIDER", "") == "ollama":
        return True

    print("⚠️  未检测到有效的 LLM 配置！")
    print()
    print("   云端部署（Streamlit Cloud）：在 Dashboard → Secrets 中添加 GROQ_API_KEY")
    print("   本地开发：复制 .env.example 为 .env，填入你的 Key")
    print()
    print("   免费获取 Groq Key：https://console.groq.com")
    return False


def init_vector_store():
    """初始化向量存储（首次运行构建 FAISS 索引）"""
    from config import get_vector_store
    from data.seed_data import build_search_texts

    vs = get_vector_store()
    if vs.is_empty():
        print("🔨 首次运行：构建知识库向量索引...")
        items = build_search_texts()
        vs.add_texts(
            texts=[it["text"] for it in items],
            metadatas=[{"id": it["id"], **it["metadata"]} for it in items],
        )
        print(f"✅ 索引构建完成：{len(items)} 条记录")
        print(f"   索引位置: data/faiss_index/")
        print(f"   嵌入模式: {vs.embedding_mode}")
    else:
        print(f"✅ 向量索引已就绪（{vs.embedding_mode}）")
    return vs


def main():
    print("=" * 56)
    print("  💧 水厂报修智能分类系统 v1.0")
    print("  Water Plant Repair Report Classification Agent")
    print("=" * 56)
    print()

    if not check_env():
        print("\n❌ 环境检查未通过")
        sys.exit(1)
    print("✅ 环境配置检查通过")

    init_vector_store()

    print("\n🚀 启动 Streamlit → http://localhost:8501")
    import subprocess
    subprocess.run([
        "streamlit", "run",
        str(Path(__file__).parent / "ui" / "app.py"),
        "--server.port", "8501",
        "--browser.serverAddress", "localhost",
    ])


if __name__ == "__main__":
    main()
