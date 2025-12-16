"""
etl/pipeline.py
高级 ETL 流水线 (集成版)。
"""

import os
import uuid
import pandas as pd  # ✅ 新增：引入 pandas 处理 Excel
from typing import Dict, List, Any

# 引入所有组件
try:
    from ingestion.tender_parser import TenderDocParser, IndexableChunk # ✅ 确保引入了 IndexableChunk
    from etl.text_cleaner import TextCleaner
    from etl.vector_store import VectorStoreManager
    from etl.deduplication import DeduplicationService
    from etl.metadata_extractor import IntelligentMetadataExtractor
    from ingestion.metadata_manager import ProjectRegistry
    from utils.graph_manager import GraphManager
except ImportError as e:
    print(f"⚠️ 模块导入失败: {e}")

class ETLPipeline:
    def __init__(self, deepseek_api_key: str):
        self.api_key = deepseek_api_key
        self.dedup = DeduplicationService()
        self.meta_extractor = IntelligentMetadataExtractor(api_key=deepseek_api_key)
        self.cleaner = TextCleaner()
        self.vector_store = VectorStoreManager()
        self.registry = ProjectRegistry()

        try:
            self.graph_manager = GraphManager()
            print("🕸️ GraphManager 连接成功")
        except Exception as e:
            print(f"⚠️ GraphManager 初始化失败 (非阻断): {e}")
            self.graph_manager = None

    # ✅ 最终完美版：既保留文本格式，又支持智能统计
    def _parse_excel(self, file_path: str, meta: Dict) -> List[IndexableChunk]:
        chunks = []
        try:
            print(f"📊 [Parser] 正在解析 Excel (完美版): {file_path}")
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                # 1. 核心改动：强制所有数据按“文本”读取，确保手机号/编号不走样
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str, keep_default_na=False)

                stats_desc = []
                total_rows = len(df)

                # --- 🧠 智能列分析 ---
                for col in df.columns:
                    # 跳过空列
                    if df[col].str.strip().eq("").all():
                        continue

                    series = df[col]

                    # 尝试转换为数值进行分析 (errors='coerce' 会把非数字变 NaN)
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    valid_count = numeric_series.count()

                    # 尝试转换为日期
                    try:
                        date_series = pd.to_datetime(series, errors='coerce')
                        valid_date_count = date_series.count()
                    except:
                        valid_date_count = 0

                    # 🔍 判定 A: 这一列主要是数字 (有效数字占比 > 80% 且不是纯 ID)
                    if valid_count > total_rows * 0.8 and series.nunique() > 10:
                        _min = numeric_series.min()
                        _max = numeric_series.max()
                        _mean = numeric_series.mean()
                        _sum = numeric_series.sum()
                        stats_desc.append(
                            f"【数值统计】'{col}': 范围[{_min} ~ {_max}], 总和={_sum:,.2f}, 均值={_mean:,.2f}"
                        )

                    # 🔍 判定 B: 这一列主要是日期
                    elif valid_date_count > total_rows * 0.8:
                        _start = date_series.min().strftime('%Y-%m-%d')
                        _end = date_series.max().strftime('%Y-%m-%d')
                        stats_desc.append(f"【时间跨度】'{col}': 从 {_start} 到 {_end}")

                    # 🔍 判定 C: 文本/分类 (排除全是数字的情况，避免把金额当分类统计)
                    elif valid_count < total_rows * 0.5:
                        str_series = series.astype(str).str.strip()
                        # 过滤掉空字符串
                        str_series = str_series[str_series != ""]
                        unique_count = str_series.nunique()

                        if unique_count <= 30:
                            counts = str_series.value_counts()
                            stats_str = ", ".join([f"{k}:{v}个" for k, v in counts.items()])
                            stats_desc.append(f"【分布统计】'{col}': {stats_str}")
                        elif unique_count < total_rows * 0.8:
                            top10 = str_series.value_counts().head(10)
                            stats_str = ", ".join([f"{k}:{v}个" for k, v in top10.items()])
                            stats_desc.append(f"【高频统计】'{col}' (前10名): {stats_str}, 以及其他...")

                # 生成摘要切片
                if stats_desc:
                    summary_text = (
                            f"【表格全景统计-{sheet_name}】\n"
                            f"数据总行数: {total_rows} 行\n"
                            "以下是关键字段的自动分析结果：\n" +
                            "\n".join(stats_desc)
                    )
                    chunks.append(IndexableChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=summary_text,
                        metadata={
                            "source_file": meta.get("source_file", ""),
                            "project_name": meta.get("project_name", ""),
                            "category": "统计摘要",
                            "page": sheet_name,
                            "type": "summary"
                        }
                    ))

                # --- 行数据转换 (现在 df 本身就是 str，直接用即可，无需再转换) ---
                for index, row in df.iterrows():
                    row_items = []
                    for col in df.columns:
                        val = str(row[col]).strip()
                        if val and val.lower() not in ['nan', 'none', '', 'null']:
                            # 简单清洗
                            if val.endswith(" 00:00:00"): val = val.replace(" 00:00:00", "")
                            row_items.append(f"{col}: {val}")

                    if row_items:
                        content_str = f"【表格数据-{sheet_name}】 " + "; ".join(row_items)
                        chunks.append(IndexableChunk(
                            chunk_id=str(uuid.uuid4()),
                            content=content_str,
                            metadata={
                                "source_file": meta.get("source_file", ""),
                                "project_name": meta.get("project_name", ""),
                                "category": meta.get("category", "表格数据"),
                                "page": sheet_name,
                                "type": "table"
                            }
                        ))
            return chunks
        except Exception as e:
            print(f"❌ Excel 解析失败: {e}")
            return []

    def process_file(self, file_path: str, use_advanced_mode: bool = True, force_update: bool = False,
                     original_filename: str = None,
                     user_project: str = None,
                     user_tag: str = None
                     ) -> Dict[str, Any]:

        current_file_name = os.path.basename(file_path)
        display_name = original_filename if original_filename else current_file_name

        result = {"file": display_name, "status": "pending", "chunks": 0, "msg": ""}
        print(f"\n🚀 [Pipeline] 启动: {display_name} (增强模式: {use_advanced_mode})")

        # --- Step 1: 查重 ---
        if not force_update and self.dedup.is_processed(file_path):
            result["status"] = "skipped"
            result["msg"] = "文件内容指纹已存在"
            print(f"⏭️ {display_name} 已存在，跳过。")
            return result

        # --- Step 2: AI 提取元数据 ---
        meta = {}
        if use_advanced_mode:
            try:
                meta = self.meta_extractor.extract(file_path)
                print(f"🧠 [Metadata] AI提取结果: {meta}")
            except Exception as e:
                print(f"⚠️ AI 元数据提取失败 ({e})，回退到默认设置")
                meta = {}

        # 强制覆盖元数据
        meta["source_file"] = display_name
        if user_project:
            meta["project_name"] = user_project
        elif "project_name" not in meta or not meta["project_name"]:
            meta["project_name"] = os.path.splitext(display_name)[0]

        if user_tag:
            meta["category"] = user_tag

        self.registry.register_project(meta.get("project_name", "Unknown"), meta)

        # --- Step 3: 解析 (Parsing) ---
        chunks = []
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            # ✅ [修改]：分支判断，支持 Excel
            if file_ext in ['.xlsx', '.xls']:
                chunks = self._parse_excel(file_path, meta)
            else:
                # 原有的 Word/PDF 解析逻辑
                parser = TenderDocParser(project_info=meta, use_advanced_mode=use_advanced_mode)
                chunks = parser.parse_file(file_path)

        except Exception as e:
            result["status"] = "error"
            result["msg"] = f"解析失败: {str(e)}"
            return result

        if not chunks:
            result["status"] = "warning"
            result["msg"] = "未提取到有效内容"
            return result

        # --- Step 4: 清洗 & ID 生成 ---
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            # 只有非 Excel 的才需要深度清洗 (Excel 已经是结构化文本了)
            if chunk.metadata.get("type") != "table":
                chunk.content = self.cleaner.clean(chunk.content)

            if chunk.metadata.get("type") == "table" or len(chunk.content) > 5:
                if original_filename:
                    chunk.metadata["source_file"] = original_filename
                chunk.metadata["project_name"] = meta.get("project_name", "Unknown")
                if "category" in meta:
                    chunk.metadata["category"] = meta["category"]

                unique_str = f"{display_name}_{i}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
                chunk.metadata["chunk_id"] = chunk_id

                cleaned_chunks.append(chunk)

        # --- Step 5: 入库 ---
        if cleaned_chunks:
            self.vector_store.add_chunks(cleaned_chunks)

            if self.graph_manager:
                try:
                    graph_data = [{
                        "content": c.content,
                        "page": c.metadata.get("page", 0),
                        "chunk_id": c.metadata["chunk_id"]
                    } for c in cleaned_chunks]

                    self.graph_manager.create_document_structure(
                        filename=display_name,
                        project=meta.get("project_name", "Unknown"),
                        chunks=graph_data
                    )
                    print(f"🕸️ [Graph] 已构建 {len(graph_data)} 个节点的图谱链")
                except Exception as e:
                    print(f"⚠️ 图谱写入异常 (不影响向量库): {e}")

            self.dedup.mark_as_processed(file_path)
            result["status"] = "success"
            result["chunks"] = len(cleaned_chunks)
            result["msg"] = "入库成功"
            print(f"✅ [Success] {display_name} 处理完成，生成 {len(cleaned_chunks)} 个切片。")
        else:
            result["status"] = "warning"
            result["msg"] = "清洗后无有效数据"

        return result
