"""
evaluation/run_eval.py
功能：基于 RAGAS 框架自动化评估 RAG 系统的各项指标。
指标：
1. Faithfulness (忠实度): 答案是否未编造？
2. Answer Relevancy (答案相关性): 回答是否切题？
3. Context Precision (上下文精确度): 检索到的内容是否有用？
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 设置 Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 1. 配置 RAGAS 使用 DeepSeek 模型作为"裁判"
# RAGAS 默认用 GPT-4 打分，我们需要把它换成 DeepSeek
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)

# 2. 配置 Embedding (用于计算相关性分数)
# 使用本地模型，避免调用 OpenAI Embedding 产生费用
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

def create_test_data():
    """
    手动定义或生成测试数据集。
    格式要求：
    - question: 问题
    - answer: RAG生成的回答
    - contexts: 检索到的原文片段 (List[str])
    - ground_truth: 标准答案 (人工撰写或由 GPT-4 生成)
    """
    # 这里模拟一次 RAG 运行的结果
    # 在实际工程中，你应该写个循环，跑一遍 rag_service，把结果存下来

    data_samples = {
        'question': ['土方开挖的边坡比例是多少？'],
        'answer': ['根据规范，土方开挖的边坡比例应控制在 1:0.5。'], # RAG 生成的
        'contexts': [['土方开挖采用自上而下... 开挖边坡严格按照 1:0.5 控制...']], # 检索到的
        'ground_truth': ['土方开挖边坡应为 1:0.5。'] # 标准答案
    }

    return Dataset.from_dict(data_samples)

def run_evaluation():
    print("🚀 [Eval] 开始 RAGAS 评估...")
    dataset = create_test_data()

    # 运行评估
    # 传入 judge_llm 让 DeepSeek 充当裁判
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=judge_llm,
        embeddings=embeddings
    )

    print("\n📊 评估结果:")
    print(results)

    # 导出为 Excel
    df = results.to_pandas()
    df.to_csv("rag_evaluation_report.csv", index=False)
    print("✅ 报告已生成: rag_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()
