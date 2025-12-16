"""
utils/bm25_manager.py
功能：基于文件的 BM25 持久化索引管理器
特点：按项目隔离存储，支持增量更新，支持多项目并发检索
"""
import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any


class BM25Persistence:
    def __init__(self, index_dir="bm25_indices"):
        self.index_dir = index_dir
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

    def _get_file_paths(self, project_name):
        # 为了文件系统安全，处理一下文件名
        safe_name = "".join([c for c in project_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        base_path = os.path.join(self.index_dir, safe_name)
        return f"{base_path}_model.pkl", f"{base_path}_data.pkl"

    def _tokenize(self, text: str) -> List[str]:
        # 使用 jieba 搜索引擎模式分词
        return list(jieba.cut_for_search(text))

    def update_project_index(self, project_name: str, new_chunks: List[Dict]):
        """
        [入库调用] 更新指定项目的索引（支持去重/覆盖更新）
        """
        model_path, data_path = self._get_file_paths(project_name)

        # 1. 加载现有数据
        existing_data = []
        if os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as f:
                    existing_data = pickle.load(f)
            except Exception as e:
                print(f"⚠️ 加载旧索引数据失败: {e}，将重建索引。")

        # ✅ [新增] 2. 严谨的去重合并逻辑 (Upsert Strategy)
        # 使用字典 key 的唯一性来自动去重
        data_map = {}

        # 2.1 先载入旧数据
        for item in existing_data:
            # 优先使用 chunk_id 作为主键
            c_id = item.get('metadata', {}).get('chunk_id')
            if c_id:
                data_map[c_id] = item
            else:
                # 兼容性处理：万一旧数据没 ID，用内容哈希兜底 (极少情况)
                content_hash = hash(item.get('content', ''))
                data_map[f"hash_{content_hash}"] = item

        # 2.2 再载入新数据 (如果 ID 相同，新数据会覆盖旧数据)
        for item in new_chunks:
            c_id = item.get('metadata', {}).get('chunk_id')
            if c_id:
                data_map[c_id] = item  # 覆盖！
            else:
                # 理论上 pipeline 保证了肯定有 ID，这里是双保险
                content_hash = hash(item.get('content', ''))
                data_map[f"hash_{content_hash}"] = item

        # 2.3 转回列表
        full_data = list(data_map.values())

        print(
            f"📊 数据合并报告: 旧数据 {len(existing_data)} 条 + 新数据 {len(new_chunks)} 条 -> 去重后总量 {len(full_data)} 条")

        if not full_data:
            return

        # 3. 重新构建 BM25 索引 (保持不变)
        print(f"🔄 正在重建项目 '{project_name}' 的 BM25 索引...")
        tokenized_corpus = [self._tokenize(doc['content']) for doc in full_data]
        bm25 = BM25Okapi(tokenized_corpus)

        # 4. 持久化保存 (保持不变)
        with open(model_path, 'wb') as f:
            pickle.dump(bm25, f)
        with open(data_path, 'wb') as f:
            pickle.dump(full_data, f)

        print(f"✅ BM25 索引已更新并保存: {project_name}")

    def search(self, query: str, projects: List[str], top_k=3) -> List[Any]:
        """
        [检索调用] 在指定项目列表中搜索
        """
        results = []
        tokenized_query = self._tokenize(query)

        # 遍历所有涉及的项目
        # 优化：如果是“所有项目”，这里需要遍历目录下所有文件（暂略，建议前端限制必须选项目）
        target_projects = projects if isinstance(projects, list) else [projects]

        for proj in target_projects:
            if proj == "所有项目": continue  # 暂不支持全库 BM25，太慢

            model_path, data_path = self._get_file_paths(proj)
            if not os.path.exists(model_path):
                continue

            try:
                # 加载索引 (优化：未来可以用 LRU Cache 缓存到内存)
                with open(model_path, 'rb') as f:
                    bm25 = pickle.load(f)
                with open(data_path, 'rb') as f:
                    corpus_data = pickle.load(f)

                # 获取分数
                scores = bm25.get_scores(tokenized_query)
                # 排序取 Top-K
                top_n = bm25.get_top_n(tokenized_query, corpus_data, n=top_k)

                results.extend(top_n)

            except Exception as e:
                print(f"⚠️ 检索项目 {proj} 失败: {e}")

        return results
