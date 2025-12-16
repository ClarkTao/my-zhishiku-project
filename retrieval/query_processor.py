"""
retrieval/query_processor.py
功能：Query Rewriting + Metadata Extraction (Self-Querying)
"""
import json
import os
from openai import Client

class QueryProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = Client(api_key=self.api_key, base_url="https://api.deepseek.com")

    def rewrite(self, user_query: str, history: list = None) -> tuple:
        """
        重写查询并提取过滤条件
        :return: (rewritten_query: str, filters: dict)
        """
        # 构建 Prompt
        history_text = ""
        if history:
            history_text = f"【对话历史】\n{history[-4:]}\n"

        # --- 痛点3：让 LLM 同时提取 Metadata Filters ---
        prompt = f"""
        你是一个智能检索引擎的预处理模块。请分析用户的输入。
        
        {history_text}
        
        【用户当前问题】: "{user_query}"
        
        【任务】
        1. **Query Rewrite**: 将问题重写为适合向量检索的专业关键词（补全主语、去口语化）。
        2. **Filter Extraction**: 判断用户是否指定了文档范围（如“大坝”、“厂房”、“地质报告”）。
           - 如果有，生成 Filter: {{ "source_file": {{ "$contains": "关键词" }} }}
           - 如果无，Filter 为 {{}}
           
        【输出格式 (JSON)】
        {{
            "query": "重写后的查询字符串",
            "filter": {{ ... }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, # 强制 JSON 模式
                temperature=0.1
            )
            res_content = response.choices[0].message.content
            res_json = json.loads(res_content)

            new_q = res_json.get("query", user_query)
            filters = res_json.get("filter", {})

            print(f"🔄 [Query] '{user_query}' -> '{new_q}' | Filter: {filters}")
            return new_q, filters

        except Exception as e:
            print(f"⚠️ Query解析失败: {e}")
            return user_query, {}
