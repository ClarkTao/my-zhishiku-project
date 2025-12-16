import os
from etl.pipeline import AdvancedETLPipeline

# 这里填入您的 DeepSeek API Key (或从环境变量读取)
API_KEY = "sk-354f38a91a674171bdf3653f9bddae36"

def main():
    # 初始化高级流水线
    pipeline = AdvancedETLPipeline(deepseek_api_key=API_KEY)

    # 指定要处理的文件
    file_path = "data/某水库除险加固工程_技术标.pdf"

    # 运行
    pipeline.run(file_path)

if __name__ == "__main__":
    main()
