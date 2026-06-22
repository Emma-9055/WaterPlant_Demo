"""
向量存储抽象层 —— Dify 切换点

通过工厂函数 create_vector_store() 获取后端实例。
默认使用 FAISS 本地向量库；设置 VECTOR_STORE_BACKEND=dify 可切换到 Dify 知识库 API。
"""
from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """向量存储抽象基类 —— 所有后端（FAISS / Dify / Chroma）均需实现此接口"""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        向量相似度检索。
        返回: [{"id": str, "text": str, "score": float, "metadata": dict}, ...]
        """
        ...

    @abstractmethod
    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """批量添加文本到向量库"""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 ID 删除文本"""
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        """向量库是否为空"""
        ...


def create_vector_store(backend: str = "faiss") -> VectorStoreBase:
    """
    向量存储工厂函数。
    由 VECTOR_STORE_BACKEND 环境变量决定后端：
    - "faiss": 本地 FAISS + sentence-transformers（默认）
    - "dify":  Dify 知识库 API（需先实现 dify_impl.py）
    """
    if backend == "faiss":
        from vector_store.faiss_impl import FAISSVectorStore
        return FAISSVectorStore()
    elif backend == "dify":
        raise NotImplementedError(
            "Dify 后端尚未实现。请创建 vector_store/dify_impl.py，"
            "实现 VectorStoreBase 接口后，将 VECTOR_STORE_BACKEND 设为 dify。"
        )
    else:
        raise ValueError(f"未知向量存储后端: {backend}，可选: faiss, dify")
