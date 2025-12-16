"""
test_rag.py
测试 RAG 生成效果
"""
import os
from generation.rag_service import DeepSeekRAGService

# 填入您的 Key
API_KEY = "sk-354f38a91a674171bdf3653f9bddae36"


def main():
    try:
        service = DeepSeekRAGService(api_key=API_KEY)
    except Exception as e:
        print(e)
        return

    # 模拟用户提问
    question = "请参考之前的项目，编写一份关于土方开挖的施工工艺流程，要求包含边坡控制参数。"
    filter_tag = None

    print(f"\n🙋‍♂️ 提问: {question}\n")
    print("🤖 DeepSeek 正在思考...\n")

    # 获取流式响应
    full_answer = ""
    for event in service.chat_stream(question, project_filter=filter_tag):

        if event["type"] == "sources":
            print(f"📚 已找到 {len(event['data'])} 份参考资料")
            for ref in event['data']:
                print(f"   - {ref['source']}")
            print("-" * 30)

        elif event["type"] == "text":
            token = event["data"]
            print(token, end="", flush=True)  # 像打字机一样输出
            full_answer += token

        elif event["type"] == "error":
            print(f"\n❌ 发生错误: {event['data']}")

    print("\n\n✅ 回答结束。")


if __name__ == "__main__":
    main()
