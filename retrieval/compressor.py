"""
retrieval/compressor.py
功能：Context Compression (上下文压缩)。
作用：从检索到的长文档块中，精准提取与问题相关的句子，去除噪音。
"""
import os
from openai import Client
from typing import List, Dict


class ContextCompressor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = Client(api_key=self.api_key, base_url="https://api.deepseek.com")

    def compress(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        使用 LLM 提炼关键信息。
        :return: 压缩后的纯文本上下文
        """
        if not retrieved_chunks:
            return ""

        # 拼接原始内容，但保留来源标记
        raw_context = ""
        for i, item in enumerate(retrieved_chunks):
            raw_context += f"--- 文档片段 {i + 1} (ID: {item.get('id')}) ---\n{item['content']}\n\n"

        prompt = f"""
        请阅读以下检索到的文档片段，针对问题“{query}”，**提取并精简**出有用的信息。

        【要求】
        1. 去除与问题无关的废话、页眉页脚、乱码。
        2. 保留关键数据（数字）、工艺步骤、规范要求。
        3. **保留原文的引用ID** (如：[文档片段 1])，不要合并不同来源的信息。
        4. 如果片段完全无关，请忽略。

        【待处理文档】
        {raw_context[:3000]} (已截断)

        【输出格式】
        [Ref: 1] ...关键内容...
        [Ref: 2] ...关键内容...
        """

        try:
            # 使用较快的模型或 deepseek-chat
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            compressed_text = response.choices[0].message.content.strip()
            # print(f"🤏 [Compression] 原始内容长度 {len(raw_context)} -> 压缩后 {len(compressed_text)}")
            return compressed_text
        except Exception as e:
            print(f"⚠️ 压缩失败: {e}")
            return raw_context  # 降级：返回原始内容
