"""
=== 核心模块: etl/graph_engine.py (企业级增强版) ===
"""
import json
import ast
import re
import time
import logging
import traceback
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraphEngine:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.1, # 保持低温，稳定输出
            max_tokens=4000,
            request_timeout=60
        )

    def generate_graph_data(self, text_content: str, custom_schema: str = None):
        """
        生成图谱数据 (支持自动重试与自定义Schema)
        """
        if not text_content:
            return {"nodes": [], "edges": []}

        # 1. Schema 定义 (支持扩展)
        default_schema = """
        【节点类型】
        - 项目    (核心项目)
        - 组织    (业主、乙方、监管方)
        - 时间    (里程碑、截止日)
        - 风险    (合规风险、技术难点)
        - 规范    (标准、法律条款)
        - 资源    (资金、设备、人员)

        【关系类型】
        - 负责    (组织 -> 项目/资源)
        - 依赖    (项目 -> 资源/资质)
        - 约束于  (项目 -> 规范)
        - 时间为  (任务 -> 时间点)
        - 存在风险(项目 -> 风险)
        """
        schema = custom_schema if custom_schema else default_schema

        # 2. Prompt 增强：加入 Few-Shot 示例
        prompt = ChatPromptTemplate.from_template("""
        你是一个工程知识图谱构建专家。请阅读文档摘要，构建可视化的关系网络。

        【Schema 约束】
        {schema}

        【文档内容】
        {text}

        【输出要求】
        1. **节点ID中文化**：严禁使用 "R1", "Proj_01" 等代号，必须使用 "三峡大坝", "2025年" 等自然语言。
        2. **关系中文化**：必须使用 Schema 定义的中文关系词。
        3. **格式严格**：输出标准 JSON，包含 nodes 和 edges。

        【输出示例】
        {{
            "nodes": [
                {{"id": "长江电力", "type": "组织", "desc": "项目业主单位"}},
                {{"id": "2025年完工", "type": "时间", "desc": "合同竣工日期"}}
            ],
            "edges": [
                {{"source": "长江电力", "target": "2025年完工", "relation": "要求"}}
            ]
        }}

        请直接输出 JSON 数据：
        """)

        # 3. 智能截取 (保留开头和结尾的关键信息，中间截断)
        if len(text_content) > 12000:
            safe_text = text_content[:8000] + "\n...\n" + text_content[-4000:]
        else:
            safe_text = text_content

        # 4. 调用与重试机制
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                chain = prompt | self.llm | StrOutputParser()
                logger.info(f"🕸️ [Graph] DeepSeek 调用中 (尝试 {attempt+1}/{max_retries+1})...")

                raw_response = chain.invoke({"schema": schema, "text": safe_text})

                # 5. 健壮解析
                graph_json = self._parse_response_robustly(raw_response)

                # 6. 后处理与校验
                return self._post_process_graph(graph_json)

            except Exception as e:
                logger.error(f"❌ 第 {attempt+1} 次生成失败: {e}")
                if attempt < max_retries:
                    time.sleep(2) # 避让策略
                else:
                    traceback.print_exc()
                    return {"nodes": [], "edges": []}

    def _parse_response_robustly(self, text: str):
        """
        三级解析策略：正则提取 -> JSON解析 -> AST解析
        """
        text = text.strip()

        # 策略 A: 尝试正则提取 JSON 部分
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            clean_text = match.group()
        else:
            clean_text = text # 兜底

        # 策略 B: 标准 JSON
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # 策略 C: Python AST (处理单引号、尾随逗号)
        try:
            return ast.literal_eval(clean_text)
        except:
            raise ValueError("无法解析为 JSON 或 Python 字典")

    def _post_process_graph(self, data):
        """
        后处理：去重、拓扑校验、度计算、样式注入
        """
        if not isinstance(data, dict): return {"nodes": [], "edges": []}

        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])

        # 1. 节点去重与字典化
        valid_nodes_map = {}
        for node in raw_nodes:
            nid = str(node.get("id", "")).strip()
            # 过滤无效节点
            if not nid or len(nid) < 2: continue

            # 如果ID已存在，保留信息量更全的那个 (此处简化为保留第一个)
            if nid not in valid_nodes_map:
                valid_nodes_map[nid] = {
                    "id": nid,
                    "type": node.get("type", "资源"),
                    "desc": node.get("desc", "")
                }

        # 2. 边的一致性校验 (剔除悬空边)
        valid_edges = []
        node_degree = Counter() # 用于计算度中心性

        for edge in raw_edges:
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            rel = edge.get("relation", "关联")

            # 核心校验：两端节点必须都存在
            if src in valid_nodes_map and tgt in valid_nodes_map and src != tgt:
                valid_edges.append({
                    "source": src,
                    "target": tgt,
                    "label": rel, # 边上的文字
                    "color": "#cccccc",
                    "font": {"align": "middle", "size": 12}
                })
                # 统计度
                node_degree[src] += 1
                node_degree[tgt] += 1

        # 3. 节点样式注入 (根据度动态调整大小)
        color_map = {
            "项目": "#005bea", "组织": "#00d2ff", "时间": "#f9a825",
            "风险": "#ff4b4b", "规范": "#2e7d32", "资源": "#6c757d"
        }

        final_nodes = []
        for nid, n_data in valid_nodes_map.items():
            degree = node_degree.get(nid, 0)
            # 基础大小 25，每多一个连接 +2，最大 60
            size = min(60, 25 + degree * 3)
            # 项目根节点特殊放大
            if n_data["type"] == "项目": size = 50

            final_nodes.append({
                "id": nid,
                "label": nid, # 显式 Label
                "title": n_data["desc"] or nid, # Tooltip
                "color": color_map.get(n_data["type"], "#999999"),
                "size": size,
                "font": {"size": 16 if size > 30 else 14, "color": "black", "face": "arial"}
            })

        logger.info(f"✅ 图谱构建完成: {len(final_nodes)} 节点, {len(valid_edges)} 边")
        return {"nodes": final_nodes, "edges": valid_edges}
