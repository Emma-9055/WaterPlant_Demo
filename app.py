"""
水厂报修智能分类系统 —— 入口文件

首次运行自动构建 FAISS 向量索引，然后启动 Streamlit UI。

使用方法:
    1. 复制 .env.example 为 .env，填入 ANTHROPIC_API_KEY
    2. pip install -r requirements.txt
    3. python app.py
"""
import sys
import os
from pathlib import Path


def check_env():
    """检查必要的环境配置"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("⚠️  未找到 .env 文件！")
        print("   请复制 .env.example 为 .env，然后填入你的 API Key:")
        print("   copy .env.example .env")
        print()
        print("   最低配置: ANTHROPIC_API_KEY=sk-ant-...")
        return False

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "sk-ant-...":
        print("⚠️  未配置 ANTHROPIC_API_KEY！")
        print("   请编辑 .env 文件，将 ANTHROPIC_API_KEY 设为你的真实 API Key。")
        return False

    return True


def init_vector_store():
    """初始化向量存储：首次运行时构建 FAISS 索引"""
    from config import get_vector_store
    from data.seed_data import build_search_texts

    vs = get_vector_store()
    if vs.is_empty():
        print("🔨 首次运行：正在构建知识库向量索引...")
        items = build_search_texts()
        vs.add_texts(
            texts=[it["text"] for it in items],
            metadatas=[{"id": it["id"], **it["metadata"]} for it in items],
        )
        print(f"✅ 索引构建完成：{len(items)} 条记录（{sum(1 for it in items if it['type']=='case')} 条案例 + {sum(1 for it in items if it['type']=='doc')} 篇文档）")
        print(f"   索引位置: data/faiss_index/")
    else:
        print("✅ 向量索引已存在，跳过构建")
    return vs


def main():
    print("=" * 60)
    print("  💧 水厂报修智能分类系统 v1.0")
    print("  Water Plant Repair Report Classification Agent")
    print("=" * 60)
    print()

    # 1. 环境检查
    if not check_env():
        print("\n❌ 环境检查未通过，请按提示配置后重新运行。")
        sys.exit(1)
    print("✅ 环境配置检查通过")

    # 2. 初始化向量存储
    init_vector_store()

    # 3. 启动 Streamlit
    print("\n🚀 启动 Streamlit UI...")
    import subprocess
    subprocess.run([
        "streamlit", "run",
        str(Path(__file__).parent / "ui" / "app.py"),
        "--server.port", "8501",
        "--browser.serverAddress", "localhost",
    ])


if __name__ == "__main__":
    main()
