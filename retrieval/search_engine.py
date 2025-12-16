"""
retrieval/search_engine.py
检索层核心逻辑 (终极版: Vector + Metadata Filter + BM25 + Rerank)
修复：
1. Metadata Filter: 正确传递 filter 到 ChromaDB
2. Hybrid Search: 在召回结果集上执行 BM25 关键词加权
3. Data Format: 正确处理父文档上下文和表格数据
"""

import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

# 引入向量库管理
try:
    from etl.vector_store import VectorStoreManager
except ImportError:
    # 路径兼容处理
    import sys
    sys.path.append("..")
    from etl.vector_store import VectorStoreManager

class TenderRetriever:
    _reranker_instance = None # 单例缓存

    def __init__(self):
        # 1. 连接向量数据库 (复用 etl/vector_store.py 的逻辑)
        self.vector_store = VectorStoreManager()

        # 2. 初始化 Reranker (单例模式，防止内存溢出)
        if TenderRetriever._reranker_instance is None:
            print("⚖️ [Retriever] 加载 BGE-Reranker 模型...")
            try:
                # 使用 BAAI/bge-reranker-base (效果比 nice, 速度适中)
                TenderRetriever._reranker_instance = CrossEncoder('BAAI/bge-reranker-base')
            except Exception as e:
                print(f"⚠️ Reranker 加载失败: {e}，将仅使用向量检索。")
                TenderRetriever._reranker_instance = None

        self.reranker = TenderRetriever._reranker_instance

    def _apply_bm25_score(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        在向量召回的候选集上，叠加 BM25 关键词分数
        (解决 '混合检索开关失效' 问题)
        """
        if not candidates: return []

        # 分词
        tokenized_query = list(jieba.cut(query))
        corpus = [list(jieba.cut(doc['content'])) for doc in candidates]

        # 计算 BM25
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        # 归一化并叠加分数
        # 注意：Vector score 通常在 0-1 之间 (Cosine)，BM25 分数可能很大
        # 这里做一个简单的加权：Final = Vector + (BM25 * 0.05)
        # 目的不是替代向量，而是让包含精确关键词的结果排名略微靠前
        max_bm25 = max(scores) if scores.any() else 1.0

        for i, doc in enumerate(candidates):
            bm25_norm = scores[i] / max_bm25 if max_bm25 > 0 else 0
            doc['score'] = doc['score'] + (bm25_norm * 0.3) # 0.3 是混合权重系数
            doc['bm25_score'] = scores[i]

        # 重新排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def search(self, query: str, top_k: int = 6, project_type: Dict = None, use_hybrid: bool = True) -> List[Dict]:
        """
        执行全流程检索
        :param query: 用户问题
        :param top_k: 最终返回数量
        :param project_type: 过滤条件 (如 {'type': '水库'}) -> 解决自查询失效
        :param use_hybrid: 是否开启混合检索 -> 解决开关失效
        """

        # --- 1. 构造过滤器 (解决 Self-Query 失效) ---
        # ChromaDB 的 where 参数格式要求严格
        chroma_filter = None
        if project_type:
            # 清理空值
            valid_filters = {k: v for k, v in project_type.items() if v and isinstance(v, str)}
            if valid_filters:
                # 如果只有一个条件
                if len(valid_filters) == 1:
                    chroma_filter = valid_filters
                # 如果有多个条件，需要用 $and (Chroma特定语法)
                else:
                    chroma_filter = {"$and": [{k: v} for k, v in valid_filters.items()]}

        print(f"🔍 [Retriever] Query: '{query}' | Filter: {chroma_filter}")

        # --- 2. 向量召回 (Recall) ---
        # 召回 3 倍数量，给 Reranker/BM25 留出筛选空间
        recall_k = top_k * 3

        # 直接调用 collection.query (最底层 API，避免封装导致的参数丢失)
        # 此时会自动调用 vector_store 中的 embedding model 进行 query 向量化
        query_vec = self.vector_store._generate_embeddings([query])

        raw_res = self.vector_store.collection.query(
            query_embeddings=query_vec,
            n_results=recall_k,
            where=chroma_filter # ✅ 关键：传入过滤器
        )

        if not raw_res['ids'] or not raw_res['ids'][0]:
            print("⚠️ 未找到相关文档")
            return []

        # --- 3. 格式化 & 数据清洗 (解决 Data Format Mismatch) ---
        candidates = []
        ids = raw_res['ids'][0]
        docs = raw_res['documents'][0]
        metas = raw_res['metadatas'][0]
        distances = raw_res['distances'][0]

        for i in range(len(ids)):
            meta = metas[i]
            doc_content = docs[i]

            # ✅ 父文档增强逻辑
            # 如果 metadata 里有 "full_context" 且不为空，说明这是个子切片，取父切片内容
            # 这样 LLM 就能看到完整的上下文
            display_content = meta.get('full_context')
            if not display_content:
                display_content = doc_content

            # ✅ 恢复表格标记
            # 如果内容看起来像 CSV 摘要，我们在 UI 上可能需要特殊处理
            if "【表格摘要】" in display_content:
                pass # 可以在这里加标记，目前保持原样即可

            candidates.append({
                "id": ids[i],
                "content": display_content, # 最终给 LLM 看的内容
                "metadata": meta,           # 元数据 (page, file等)
                "score": 1 - distances[i],  # 将距离转为相似度 (近似)
                "source_file": meta.get('source_file', 'Unknown')
            })

        # --- 4. 混合检索 (BM25优化) ---
        if use_hybrid:
            # 在向量召回的基础上，根据关键词匹配度微调分数
            candidates = self._apply_bm25_score(query, candidates)

        # --- 5. 重排序 (Rerank) ---
        if self.reranker:
            # 构造 [Query, Doc] 对
            pairs = [[query, c['content']] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for i, c in enumerate(candidates):
                c['score'] = float(rerank_scores[i]) # 覆盖为 Reranker 的绝对分数

            # 按 Rerank 分数最终排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            print(f"⚖️ [Rerank] 重排序完成，Top-1 得分: {candidates[0]['score']:.4f}")

        # --- 6. 返回 Top-K ---
        return candidates[:top_k]

# --- 单元测试 ---
if __name__ == "__main__":
    print("🚀 测试检索引擎...")
    engine = TenderRetriever()

    # 测试 1: 基础检索
    print("\n--- Test 1: Basic Search ---")
    res = engine.search("大坝混凝土标号", top_k=2)
    for r in res:
        print(f"[{r['score']:.2f}] {r['source_file']} : {r['content'][:50]}...")

    # 测试 2: 过滤器 (模拟 app.py 传来的参数)
    print("\n--- Test 2: With Filter (type='水库') ---")
    # 注意：确保你数据库里真有 type='水库' 的数据，否则这里是空的
    res = engine.search("大坝", top_k=2, project_type={"type": "水库"})
    print(f"Hit count: {len(res)}")
