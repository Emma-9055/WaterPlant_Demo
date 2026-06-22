"""
FAISS 向量存储实现 —— 本地向量检索，零外部 API 依赖

嵌入模型: paraphrase-multilingual-MiniLM-L12-v2 (118 MB)
- 支持中文语义相似度检索
- 首次运行自动下载模型
- 索引持久化到 ../data/faiss_index/
"""
import os
import pickle
import numpy as np
from pathlib import Path
from config import FAISS_INDEX_DIR
from vector_store import VectorStoreBase


class FAISSVectorStore(VectorStoreBase):
    """基于 FAISS + sentence-transformers 的本地向量存储"""

    def __init__(self):
        import faiss
        self.faiss = faiss

        # 加载嵌入模型
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # 兼容新旧版本 API
        try:
            self._dim = self._model.get_sentence_embedding_dimension()
        except AttributeError:
            self._dim = self._model.get_embedding_dimension()

        # 元数据存储: faiss_id → {"id": str, "text": str, "metadata": dict}
        self._meta: dict[int, dict] = {}
        self._next_id = 0

        # 加载或创建索引
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._index_path = str(FAISS_INDEX_DIR / "index.faiss")
        self._meta_path = str(FAISS_INDEX_DIR / "metadata.pkl")

        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = self.faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                saved = pickle.load(f)
                self._meta = saved["meta"]
                self._next_id = saved["next_id"]
        else:
            # 创建空索引（使用 L2 距离，32 条数据用精确搜索即可）
            self._index = self.faiss.IndexFlatL2(self._dim)
            self._save()

    # ---- 核心接口 ----

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        向量相似度检索。
        返回得分越高越相似（已将 L2 距离转换为 0-1 相似度分数）。
        """
        if self._index.ntotal == 0:
            return []

        query_vec = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
        distances, indices = self._index.search(query_vec, min(k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self._meta:
                continue
            meta = self._meta[idx]
            # L2 归一化向量距离范围 [0, 2]，转为 [0, 1] 相似度
            similarity = max(0.0, 1.0 - dist / 2.0)
            results.append({
                "id": meta["id"],
                "text": meta["text"],
                "score": round(similarity, 4),
                "metadata": meta.get("metadata", {}),
            })

        # 去重（同一个原始 id 只保留最高分）
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)
        return deduped[:k]

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """批量添加文本"""
        if not texts:
            return

        # 跳过已存在的相同 ID
        existing_ids = {m["id"] for m in self._meta.values()}
        to_add = []
        to_add_metas = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            text_id = meta.get("id", f"auto_{self._next_id + len(to_add)}")
            if text_id in existing_ids:
                continue
            to_add.append(text)
            to_add_metas.append(meta)

        if not to_add:
            return

        vectors = self._model.encode(to_add, normalize_embeddings=True).astype(np.float32)
        for vec, text, meta in zip(vectors, to_add, to_add_metas):
            self._index.add(vec.reshape(1, -1))
            self._meta[self._next_id] = {
                "id": meta.get("id", f"auto_{self._next_id}"),
                "text": text,
                "metadata": meta,
            }
            self._next_id += 1

        self._save()

    def delete(self, ids: list[str]) -> None:
        """删除指定 ID 的文本（重建索引）"""
        ids_set = set(ids)
        new_meta = {}
        vectors_to_keep = []

        for faiss_id, meta in self._meta.items():
            if meta["id"] not in ids_set:
                vectors_to_keep.append((faiss_id, meta))

        if len(vectors_to_keep) == len(self._meta):
            return  # 没有匹配的 ID

        # 重建索引
        new_index = self.faiss.IndexFlatL2(self._dim)
        new_meta = {}
        new_next = 0

        # 需要重新编码所有保留文本
        for _, meta in vectors_to_keep:
            vec = self._model.encode([meta["text"]], normalize_embeddings=True).astype(np.float32)
            new_index.add(vec.reshape(1, -1))
            new_meta[new_next] = meta
            new_next += 1

        self._index = new_index
        self._meta = new_meta
        self._next_id = new_next
        self._save()

    def is_empty(self) -> bool:
        return self._index.ntotal == 0

    # ---- 内部 ----

    def _save(self):
        """持久化索引和元数据"""
        self.faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump({"meta": self._meta, "next_id": self._next_id}, f)


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    from data.seed_data import build_search_texts

    store = FAISSVectorStore()
    print(f"索引状态: {'已存在' if not store.is_empty() else '空索引'}")

    # 首次运行：写入数据
    if store.is_empty():
        print("正在构建索引...")
        items = build_search_texts()
        store.add_texts(
            texts=[it["text"] for it in items],
            metadatas=[{"id": it["id"], **it["metadata"]} for it in items],
        )
        print(f"完成：写入 {len(items)} 条记录")

    # 测试搜索
    test_queries = [
        "水泵异响振动大",
        "管道漏水焊缝开裂",
        "出厂水浊度超标",
    ]
    for q in test_queries:
        print(f"\n🔍 搜索: '{q}'")
        results = store.search(q, k=3)
        for i, r in enumerate(results):
            print(f"  #{i+1} [{r['score']:.3f}] {r['text'][:80]}...")
