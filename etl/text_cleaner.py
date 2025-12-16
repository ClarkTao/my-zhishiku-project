"""
etl/text_cleaner.py
功能：修复 OCR 或 PDF 解析后的文本瑕疵
"""
import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text: return ""

        # 1. 修复中文跨行断句 (核心痛点)
        # 场景：PDF中 "施工\n方案" 应该合并为 "施工方案"
        # 规则：如果 "汉字\n汉字"，说明是排版换行，应该删掉 \n
        text = re.sub(r'([\u4e00-\u9fa5])\s*\n\s*([\u4e00-\u9fa5])', r'\1\2', text)

        # 2. 规范化空白字符 (Tab 转空格，连续空格转单一空格)
        text = text.replace('\t', ' ')
        # 注意：这里我们保留单个换行符 \n，因为段落结构对语义切分很重要
        # 只把连续的水平空格合并
        text = re.sub(r'[ \f\r\v]+', ' ', text)

        # 3. 去除不可见控制字符 (但保留换行符)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        return text.strip()
