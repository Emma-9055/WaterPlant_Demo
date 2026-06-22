"""快速验证脚本：逐模块测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 50)
print("Phase 1: 配置测试")
print("=" * 50)
from config import DATA_DIR, FAISS_INDEX_DIR
print(f"  DATA_DIR: {DATA_DIR}")
print(f"  FAISS_INDEX_DIR: {FAISS_INDEX_DIR}")
print("  ✅ config 加载正常")

print()
print("=" * 50)
print("Phase 2: 数据层测试")
print("=" * 50)
from data.seed_data import (
    PLANTS, CATEGORIES, HISTORICAL_CASES,
    KNOWLEDGE_DOCS, build_search_texts, get_plant_by_name
)
print(f"  水厂数量: {len(PLANTS)}")
print(f"  报修类别: {len(CATEGORIES)}")
print(f"  历史案例: {len(HISTORICAL_CASES)}")
print(f"  知识文档: {len(KNOWLEDGE_DOCS)}")
items = build_search_texts()
print(f"  检索条目: {len(items)}")
print(f"  案例条目: {sum(1 for it in items if it['type']=='case')}")
print(f"  文档条目: {sum(1 for it in items if it['type']=='doc')}")

# 测试模糊查找
plant = get_plant_by_name("城北")
print(f"  查找 '城北': {plant['name'] if plant else '未找到'}")
plant = get_plant_by_name("高新区")
print(f"  查找 '高新区': {plant['name'] if plant else '未找到'}")
print("  ✅ 数据层正常")

print()
print("=" * 50)
print("Phase 3: 向量存储测试")
print("=" * 50)
from vector_store import create_vector_store
vs = create_vector_store("faiss")
print(f"  索引状态: {'已存在' if not vs.is_empty() else '空索引'}")

if vs.is_empty():
    print("  正在构建索引...")
    vs.add_texts(
        texts=[it["text"] for it in items],
        metadatas=[{"id": it["id"], **it["metadata"]} for it in items],
    )
    print(f"  ✅ 索引构建完成")

# 测试搜索
results = vs.search("水泵异响振动大", k=3)
print(f"  搜索 '水泵异响振动大' → {len(results)} 条结果")
for r in results:
    print(f"    [{r['score']:.3f}] {r['text'][:60]}...")
print("  ✅ 向量存储正常")

print()
print("=" * 50)
print("Phase 4: Agent 工具测试 (跳过 LLM 调用)")
print("=" * 50)
from agent.tools import create_tools
# 工具创建需要 LLM，先跳过实际调用
print("  ✅ 工具模块导入正常")

print()
print("=" * 50)
print("Phase 5: Agent 编排测试 (跳过 LLM 调用)")
print("=" * 50)
from agent.orchestrator import RepairAgent, SYSTEM_PROMPT
print(f"  System Prompt 长度: {len(SYSTEM_PROMPT)} 字符")
print("  ✅ 编排模块导入正常")

print()
print("=" * 50)
print("全部模块验证通过！")
print("=" * 50)
print()
print("下一步: 创建 .env 文件，配置 ANTHROPIC_API_KEY，然后运行 python app.py")
