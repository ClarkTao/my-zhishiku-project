"""
etl/vector_store.py
功能：管理向量数据库 (ChromaDB)
修正：使用‘适配器模式’实现 LangChain 兼容，解决内存双倍占用和向量不一致问题。
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from typing import List, Any

# --- 引入 LangChain 基础类 ---
try:
    from langchain_community.vectorstores import Chroma as LangChainChroma
    from langchain_core.embeddings import Embeddings # 引入基类
except ImportError:
    pass

CHROMA_PATH = "chroma_db"

# ✅ 零开销适配器
# 这个类只是个“传话筒”，它不占内存，直接调用原有的 embedding_model
class LightweightEmbeddings(Embeddings):
    def __init__(self, transformer_model):
        self.model = transformer_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 直接复用 ETL 的模型进行推理，保证向量 100% 一致
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        # query 也是同理
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class VectorStoreManager:
    def __init__(self, collection_name="tender_docs"):
        print("⏳ [ETL] 正在初始化 Embedding 模型 (BGE-Small)...")
        try:
            model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5')
        except:
            model_dir = "BAAI/bge-small-zh-v1.5"

        # 1. 原生 Embedding (ETL 核心)
        self.embedding_model = SentenceTransformer(model_dir)

        print(f"⏳ [ETL] 连接向量数据库: {CHROMA_PATH}")
        # 2. 原生 Client (ETL 核心)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # --- ✅ 修改点：构建 LangChain 兼容层 (无副作用版) ---
        print("🔌 [Bridge] 正在构建 LangChain 兼容层...")
        try:
            # A. 使用适配器，而不是重新加载模型！(解决内存问题 + 一致性问题)
            # 我们把 self.embedding_model 传进去
            bridge_embeddings = LightweightEmbeddings(self.embedding_model)

            # B. 初始化 LangChain Chroma
            # 这里的 client=self.client 解决了 SQLite 锁冲突问题
            self.vector_store = LangChainChroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=bridge_embeddings
            )
            print("✅ [Bridge] LangChain VectorStore 就绪 (共享内存与连接)")

        except Exception as e:
            print(f"⚠️ LangChain 兼容层初始化失败: {e}")
            self.vector_store = None


    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 原有 ETL 逻辑保持不变
        return self.embedding_model.encode(texts, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: List[Any]):
        """
        原生入库逻辑 (保持不变)
        """
        if not chunks: return

        # 1. 预构建 Parent Map
        parent_map = {c.chunk_id: c.content for c in chunks if getattr(c, 'is_parent', False)}

        ids = []
        documents = []
        metadatas = []
        texts_to_embed = []

        for chunk in chunks:
            # 过滤超长父块
            if getattr(chunk, 'is_parent', False) and len(chunk.content) > 800:
                continue

            meta = chunk.metadata.copy()

            # 父子融合
            parent_id = getattr(chunk, 'parent_id', None)
            if parent_id and parent_id in parent_map:
                meta["full_context"] = parent_map[parent_id]
                meta["is_child"] = "True"
            else:
                meta["full_context"] = chunk.content
                meta["is_child"] = "False"

            # 清洗 Metadata
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (list, dict)):
                    clean_meta[k] = str(v)
                else:
                    clean_meta[k] = v

            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(clean_meta)
            texts_to_embed.append(chunk.content)

        if not ids:
            print("⚠️ 无有效数据入库")
            return

        # 2. 生成向量
        print(f"⚡ [ETL] 计算向量中 ({len(texts_to_embed)} 条)...")
        embeddings = self._generate_embeddings(texts_to_embed)

        # 3. 批量写入
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end]
            )
        print(f"✅ [ETL] 成功存入 {len(ids)} 条数据。")

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        """
        原生检索封装
        """
        query_vec = self._generate_embeddings([query])
        return self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            where=filters
        )
