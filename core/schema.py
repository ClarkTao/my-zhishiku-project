"""
core/schema.py
定义投标文件RAG系统的核心数据结构。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class TenderChunk:
    """
    文档切片对象。
    替代 KG 系统中的 Triple。
    """
    content: str  # 切片文本内容 (喂给 DeepSeek 的核心素材)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # --- 溯源信息 (用于引用功能) ---
    source_file: str = ""  # 来源文件名 (e.g., "XX水库施工标书.docx")
    page_number: int = 0  # 页码 (PDF/Word页码)
    section_title: str = ""  # 所属章节 (e.g., "第三章 施工组织设计 > 3.2 土方开挖")

    # --- 检索增强元数据 (用于"参考相同项目"功能) ---
    project_metadata: Dict[str, str] = field(default_factory=dict)

    # 示例: {
    #   "project_type": "水库除险加固",
    #   "location": "南方多雨区",
    #   "year": "2023",
    #   "doc_type": "技术标"
    # }

    def to_metadata_dict(self):
        """转换为向量数据库需要的 metadata 字典"""
        meta = {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "chunk_id": self.chunk_id
        }
        meta.update(self.project_metadata)
        return meta
