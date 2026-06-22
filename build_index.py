"""一键构建 FAISS 索引"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vector_store.faiss_impl import FAISSVectorStore
from data.seed_data import build_search_texts

store = FAISSVectorStore()
print(f"Embedding mode: {store.embedding_mode}")

# Rebuild (clear old)
items = build_search_texts()
# Force rebuild by clearing
while not store.is_empty():
    ids_to_del = [m["id"] for m in store._meta.values()]
    store.delete(ids_to_del[:1])

store.add_texts(
    [i["text"] for i in items],
    [{"id": i["id"], **i["metadata"]} for i in items],
)
print(f"Index built: {len(items)} docs")

# Test
r = store.search("水泵异响振动大", k=3)
print("Search '水泵异响振动大':")
for rr in r:
    print(f"  [{rr['score']:.3f}] {rr['text'][:60]}...")

r2 = store.search("水质浊度超标", k=3)
print("Search '水质浊度超标':")
for rr in r2:
    print(f"  [{rr['score']:.3f}] {rr['text'][:60]}...")

print("\nDone! Index is ready.")
