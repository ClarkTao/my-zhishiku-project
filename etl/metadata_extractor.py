"""
etl/metadata_extractor.py
功能：利用 LLM (DeepSeek) 智能分析文档前几页，提取结构化元数据。
"""

import json
import re
import os
from typing import Dict
from openai import Client # 统一使用 Client

class IntelligentMetadataExtractor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        # 初始化 DeepSeek 客户端
        self.client = Client(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def _read_cover_pages(self, file_path: str, max_pages=3) -> str:
        """读取 PDF/Word 的前 N 页文本用于分析"""
        text_buffer = []
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    # 只读前几页，防止 token 溢出
                    for i, page in enumerate(pdf.pages[:max_pages]):
                        text_buffer.append(page.extract_text() or "")
            elif ext in [".docx", ".doc"]:
                from docx import Document
                doc = Document(file_path)
                # Word 读取前 20 段作为封面内容
                for para in doc.paragraphs[:20]:
                    text_buffer.append(para.text)
        except ImportError:
            print("⚠️ 缺少 pdfplumber 或 python-docx 库，无法读取封面。")
        except Exception as e:
            print(f"⚠️ 读取封面失败: {e}")

        return "\n".join(text_buffer)

    def extract(self, file_path: str) -> Dict[str, str]:
        print(f"🧠 [AI提取] 正在分析文档元数据: {os.path.basename(file_path)}...")

        context_text = self._read_cover_pages(file_path)

        # 如果读不到内容，或者没有配置 API Key，直接回退
        if not context_text or not self.api_key:
            return self._fallback_extraction(file_path)

        prompt = f"""
        你是一个水利工程招投标专家。请从以下标书的前几页内容中，提取关键元数据。
        如果找不到某项信息，请留空。

        【待分析内容】
        {context_text[:2000]} ... (截断)

        【要求】
        请仅返回一个合法的 JSON 对象，不要包含 markdown 格式，包含以下字段：
        1. "project_name": 项目全称
        2. "province": 省份 (如: 四川省)
        3. "year": 招标年份 (格式: YYYY)
        4. "type": 工程类型 (从以下列表中选择最匹配的一个: 水库, 堤防, 河道, 泵站, 水电站, 灌区, 饮水安全)

        JSON:
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            # 清理可能的 markdown 标记
            content = content.replace("```json", "").replace("```", "")
            meta = json.loads(content)

            # 补充源文件名
            meta["source_file"] = os.path.basename(file_path)
            meta["source_method"] = "ai_extraction"

            # 确保 type 字段存在，用于后续过滤
            if "type" not in meta: meta["type"] = "其他"

            return meta

        except Exception as e:
            print(f"❌ AI 提取失败: {e}，切换到文件名匹配模式...")
            return self._fallback_extraction(file_path)

    def _fallback_extraction(self, file_path: str) -> Dict[str, str]:
        """回退逻辑：基于文件名进行正则和关键词匹配"""
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]

        meta = {
            "project_name": filename_no_ext,
            "source_file": filename,
            "source_method": "filename_rule_match",
            "type": "其他" # 默认值
        }

        # 1. 提取年份
        year_match = re.search(r'(202\d)', filename)
        if year_match: meta["year"] = year_match.group(1)

        # 2. 提取类型
        type_map = {
            "水库": "水库", "大坝": "水库", "除险": "水库",
            "堤防": "堤防", "堤": "堤防", "防洪": "堤防",
            "河道": "河道", "清淤": "河道",
            "泵站": "泵站", "电站": "水电站",
            "灌区": "灌区", "饮水": "饮水安全"
        }
        for keyword, p_type in type_map.items():
            if keyword in filename:
                meta["type"] = p_type
                break

        # 3. 提取省份
        provinces = ["四川", "云南", "贵州", "广东", "广西", "湖南", "湖北", "江西", "重庆", "河南", "河北", "新疆", "西藏"]
        for prov in provinces:
            if prov in filename:
                meta["province"] = prov
                break

        return meta
