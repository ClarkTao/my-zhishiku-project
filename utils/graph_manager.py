"""
utils/graph_manager.py
功能：Neo4j 图数据库管理器，负责构建文档的线性结构图谱
"""
from neo4j import GraphDatabase
import uuid


class GraphManager:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="deepseek_password"):
        # ✅ 正确写法
        self.driver = GraphDatabase.driver(uri, auth=("neo4j", "MyStrongPassword123"))

    def close(self):
        self.driver.close()

    def create_document_structure(self, filename, project, chunks):
        """
        核心写入逻辑：将切片像链条一样存入图数据库
        结构：(:Document)-[:CONTAINS]->(:Chunk)-[:NEXT]->(:Chunk)
        """
        with self.driver.session() as session:
            session.execute_write(self._create_chain_tx, filename, project, chunks)

    def _create_chain_tx(self, tx, filename, project, chunks):
        # 1. 创建文档节点
        query_doc = """
        MERGE (d:Document {name: $filename, project: $project})
        RETURN d
        """
        tx.run(query_doc, filename=filename, project=project)

        prev_chunk_id = None

        # 2. 循环创建切片节点，并建立 :NEXT 关系
        for i, chunk_text in enumerate(chunks):
            # 生成唯一 ID (这个 ID 也要存入 ChromaDB 以便联动)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i}"))

            query_chunk = """
            MATCH (d:Document {name: $filename})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text, c.page = $page, c.index = $index
            MERGE (d)-[:CONTAINS]->(c)
            """
            tx.run(query_chunk, filename=filename, chunk_id=chunk_id,
                   text=chunk_text['content'], page=chunk_text['page'], index=i)

            # 3. 建立“下一段”连接 (线性逻辑的关键！)
            if prev_chunk_id:
                query_link = """
                MATCH (prev:Chunk {id: $prev_id}), (curr:Chunk {id: $curr_id})
                MERGE (prev)-[:NEXT]->(curr)
                """
                tx.run(query_link, prev_id=prev_chunk_id, curr_id=chunk_id)

            prev_chunk_id = chunk_id

    def get_context_window(self, chunk_id, window_size=1):
        """
        核心读取逻辑：给定一个切片 ID，向左向右各找 window_size 个切片
        """
        query = """
        MATCH (target:Chunk {id: $chunk_id})
        // 找前文 (反向遍历 NEXT 关系)
        OPTIONAL MATCH (prev:Chunk)-[:NEXT*1..%d]->(target)
        // 找后文 (正向遍历 NEXT 关系)
        OPTIONAL MATCH (target)-[:NEXT*1..%d]->(next:Chunk)
        RETURN prev, target, next
        ORDER BY prev.index, next.index
        """ % (window_size, window_size)  # 动态插入步长

        with self.driver.session() as session:
            result = session.run(query, chunk_id=chunk_id)
            # 这里需要处理返回结果，将其合并成完整的上下文文本
            # (代码略，通常是提取 prev.text + target.text + next.text)
            return result.data()
