"""
ingestion/processors.py
功能：包含 OCR 识别、表格语义化摘要、图像描述生成等原子能力。
"""
import os
import io
import pandas as pd
from typing import List
from openai import Client

# --- 1. OCR 处理器 (基于 RapidOCR) ---
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ 未检测到 rapidocr_onnxruntime，OCR 功能将不可用。")

class OCRProcessor:
    @staticmethod
    def extract_text_from_image(image_bytes: bytes) -> str:
        if not HAS_OCR: return ""
        try:
            result, _ = ocr_engine(image_bytes)
            if not result: return ""
            text = "\n".join([line[1] for line in result])
            return f"\n> [OCR识别内容]:\n{text}\n"
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

# --- 2. [升级] 表格语义摘要处理器 ---
class TableProcessor:
    """
    旧的 Markdown 转换保留作为降级方案，
    主要推荐使用 TableSummarizer。
    """
    @staticmethod
    def table_to_markdown(table_data: List[List[str]]) -> str:
        if not table_data: return ""
        cleaned_data = []
        for row in table_data:
            cleaned_row = [str(cell).replace('\n', '<br>').replace('|', '&#124;') if cell else "" for cell in row]
            cleaned_data.append(cleaned_row)
        if not cleaned_data: return ""
        headers = cleaned_data[0]
        header_str = "| " + " | ".join(headers) + " |"
        separator_str = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_rows = []
        for row in cleaned_data[1:]:
            if len(row) < len(headers):
                row += [""] * (len(headers) - len(row))
            body_rows.append("| " + " | ".join(row[:len(headers)]) + " |")
        return f"\n\n{header_str}\n{separator_str}\n" + "\n".join(body_rows) + "\n\n"

class TableSummarizer:
    """
    针对痛点1：表格语义化
    将二维表格转换为自然语言摘要，解决多级表头和合并单元格的检索难题。
    """
    def __init__(self, api_key=None):
        self.client = Client(api_key=api_key or os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def summarize_table(self, table_data: list) -> str:
        """
        :param table_data: 二维列表 [[row1], [row2]...]
        :return: "语义摘要 + HTML结构"
        """
        if not table_data: return ""

        # 1. 转为 HTML (保留结构用于前端显示)
        try:
            df = pd.DataFrame(table_data)
            # 处理空表头情况
            if df.empty: return ""

            # 生成 HTML (用于展示)
            html_content = df.to_html(header=False, index=False, border=1)

            # 生成 CSV 文本 (用于 LLM 理解)
            csv_text = df.to_csv(header=False, index=False)
        except Exception as e:
            print(f"表格转换错误: {e}")
            return ""

        # 2. 调用 DeepSeek 生成摘要
        prompt = f"""
        请阅读下面的表格数据（CSV格式），生成一段简短的自然语言摘要。
        
        【要求】
        1. 识别表头，说明表格的主题（如“土方开挖机械配置表”）。
        2. 提取关键数据行，用自然语言描述（如“挖掘机配置了4台”）。
        3. 如果有多级表头，请将其逻辑理顺。
        4. 不要输出Markdown，直接输出纯文本摘要。

        【表格数据】
        {csv_text[:1500]} 
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )
            summary = response.choices[0].message.content.strip()
            # 返回：[语义摘要] + [原始结构]
            return f"【表格摘要】：{summary}\n\n【原始表格HTML】：\n{html_content}"
        except Exception as e:
            print(f"⚠️ 表格摘要生成失败: {e}")
            # 降级：只返回 Markdown
            return TableProcessor.table_to_markdown(table_data)

# --- 3. 图像描述生成器 ---
import base64
class ImageDescriptionService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = None
        if self.api_key:
            self.client = Client(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate_caption(self, image_bytes: bytes) -> str:
        if not self.client: return ""
        try:
            # 这里的 Prompt 仅供示例，需配合支持 Vision 的模型
            return "\n> [图片内容]: (需接入 Vision 模型 API)\n"
        except Exception:
            return ""
