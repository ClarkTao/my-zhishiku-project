"""
retrieval/reranker.py (Final)
功能：重排序模型 (Cross-Encoder)。
依赖：pip install sentencepiece
"""

from sentence_transformers import CrossEncoder
from modelscope import snapshot_download
import os

class BGEReranker:
    def __init__(self):
        print("⏳ [Rerank] 正在初始化重排序模型 (BGE-Reranker)...")

        local_cache_dir = './model_cache'
        target_model = 'Xorbits/bge-reranker-base'

        try:
            model_dir = snapshot_download(target_model, cache_dir=local_cache_dir)
        except Exception as e:
            print(f"⚠️ 下载失败 ({e})，尝试备用源...")
            try:
                model_dir = snapshot_download('BAAI/bge-reranker-base', cache_dir=local_cache_dir)
            except:
                model_dir = "BAAI/bge-reranker-base"

        print(f"   📂 模型路径: {model_dir}")

        # --- 最终配置 ---
        # 1. trust_remote_code=True: 必需。
        # 2. tokenizer_kwargs={"use_fast": False}: 强制使用慢速分词器 (依赖 sentencepiece)。
        #    这能解决 Windows 下 FastTokenizer 加载失败的 Bug。
        self.model = CrossEncoder(
            model_dir,
            max_length=512,
            trust_remote_code=True,
            tokenizer_kwargs={"use_fast": False}
        )
        print("✅ [Rerank] 重排序模型加载完成。")

    def rank(self, query: str, initial_results: list, top_k: int = 5) -> list:
        if not initial_results:
            return []

        passages = [res['content'] for res in initial_results]
        model_inputs = [[query, doc] for doc in passages]

        scores = self.model.predict(model_inputs)

        ranked_results = []
        for i, score in enumerate(scores):
            item = initial_results[i].copy()
            item['score'] = float(score)
            ranked_results.append(item)

        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results[:top_k]
