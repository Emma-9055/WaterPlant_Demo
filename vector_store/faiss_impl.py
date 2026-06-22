"""
FAISS 向量存储实现 —— 本地向量检索，零外部 API 依赖

双模式嵌入：
  Mode A: sentence-transformers（默认，384 维语义向量，中文效果好）
  Mode B: TF-IDF（轻量降级，纯 sklearn，10MB 内存，Streamlit Cloud 友好）

首次启动自动选择：尝试加载 sentence-transformers → 失败则降级到 TF-IDF
索引持久化到 data/faiss_index/
"""
import os
import pickle
import numpy as np
from pathlib import Path
from config import FAISS_INDEX_DIR
from vector_store import VectorStoreBase


class FAISSVectorStore(VectorStoreBase):
    """基于 FAISS 的本地向量存储，支持 transformer / TF-IDF 双模式"""

    def __init__(self):
        import faiss
        self.faiss = faiss

        # 元数据存储: faiss_id → {"id": str, "text": str, "metadata": dict}
        self._meta: dict[int, dict] = {}
        self._next_id = 0

        # 持久化路径
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._index_path = str(FAISS_INDEX_DIR / "index.faiss")
        self._meta_path = str(FAISS_INDEX_DIR / "metadata.pkl")

        # ---- 选择嵌入模式 ----
        self._use_transformer = False
        self._tfidf = None  # TfidfVectorizer 实例（降级模式）

        # 先检查 HF 本地缓存；缓存命中→离线秒加载，否则→直接 TF-IDF
        if self._is_hf_model_cached():
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"  # 跳过网络校验，纯离线
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    local_files_only=True,
                )
                try:
                    self._dim = self._model.get_sentence_embedding_dimension()
                except AttributeError:
                    self._dim = self._model.get_embedding_dimension()
                self._use_transformer = True
            except Exception as e:
                import sys
                print(f"  [!] Transformer load failed, fallback TF-IDF: {e}",
                      file=sys.stderr)
                self._init_tfidf()
                self._dim = 2048
        else:
            # 模型未缓存（或 HF 被墙），直接用 TF-IDF，零网络依赖
            self._init_tfidf()
            self._dim = 2048

        # ---- 加载或创建索引 ----
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._load_index()
        else:
            self._index = self.faiss.IndexFlatL2(self._dim)
            self._save()

    def _is_hf_model_cached(self) -> bool:
        """检查 HF 模型是否已完整缓存（snapshots 目录存在）"""
        cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
        )
        snapshots = os.path.join(cache_path, "snapshots")
        return os.path.isdir(snapshots) and len(os.listdir(snapshots)) > 0

    def _init_tfidf(self):
        """初始化 TF-IDF 向量器（轻量降级方案）"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(
            max_features=2048,
            analyzer="char_wb",
            ngram_range=(2, 4),
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        """对文本列表编码，返回 float32 向量数组"""
        if self._use_transformer:
            return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)
        else:
            # TF-IDF 模式
            if self._tfidf is None:
                self._init_tfidf()
            # 收集所有已索引文本用于拟合
            all_texts = [m["text"] for m in self._meta.values()]
            if all_texts:
                self._tfidf.fit(all_texts)
            else:
                self._tfidf.fit(texts)
            mat = self._tfidf.transform(texts).toarray().astype(np.float32)
            # L2 归一化
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return mat / norms

    def _load_index(self):
        """从磁盘加载索引"""
        self._index = self.faiss.read_index(self._index_path)
        with open(self._meta_path, "rb") as f:
            saved = pickle.load(f)
            self._meta = saved.get("meta", {})
            self._next_id = saved.get("next_id", 0)
        # 维度必须匹配
        if self._index.d != self._dim:
            # 嵌入模式变化，重建空索引
            self._index = self.faiss.IndexFlatL2(self._dim)
            self._meta = {}
            self._next_id = 0

    # ---- 核心接口 ----

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        向量相似度检索。
        返回得分越高越相似（已将距离转换为 0-1 相似度）。
        """
        if self._index.ntotal == 0:
            return []

        query_vec = self._embed([query])
        distances, indices = self._index.search(query_vec, min(k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self._meta:
                continue
            meta = self._meta[idx]
            # 距离 → 相似度（L2 归一化向量范围 [0, 2]）
            similarity = max(0.0, 1.0 - dist / 2.0)
            results.append({
                "id": meta["id"],
                "text": meta["text"],
                "score": round(similarity, 4),
                "metadata": meta.get("metadata", {}),
            })

        # 去重（同一原始 id 只保留最高分）
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

        vectors = self._embed(to_add)
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
        entries_to_keep = [
            (idx, meta) for idx, meta in self._meta.items()
            if meta["id"] not in ids_set
        ]

        if len(entries_to_keep) == len(self._meta):
            return  # 无匹配

        # 重建索引
        new_index = self.faiss.IndexFlatL2(self._dim)
        new_meta = {}
        new_next = 0

        texts_to_reencode = [meta["text"] for _, meta in entries_to_keep]
        vectors = self._embed(texts_to_reencode)
        for vec, (_, meta) in zip(vectors, entries_to_keep):
            new_index.add(vec.reshape(1, -1))
            new_meta[new_next] = meta
            new_next += 1

        self._index = new_index
        self._meta = new_meta
        self._next_id = new_next
        self._save()

    def is_empty(self) -> bool:
        return self._index.ntotal == 0

    @property
    def embedding_mode(self) -> str:
        """当前嵌入模式"""
        return "sentence-transformers" if self._use_transformer else "TF-IDF (lightweight)"

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
    print(f"嵌入模式: {store.embedding_mode}")
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
        print(f"\n搜索: '{q}'")
        results = store.search(q, k=3)
        for i, r in enumerate(results):
            print(f"  #{i+1} [{r['score']:.3f}] {r['text'][:80]}...")
