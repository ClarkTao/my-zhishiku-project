# 🌊 规划设计智能助手 (Intelligent Planning & Design Assistant)

本项目是一个基于大语言模型（LLM, 如 DeepSeek V3）和检索增强生成（RAG）技术的智能问答系统。它旨在成为规划设计领域的专业知识库和智能助手，能够对上传的标书、技术规范、合同、图纸等文档进行深度分析、风险识别、数据提取和智能问答。

## ✨ 主要功能

- **多项目知识库管理**: 支持创建多个独立的项目空间，上传和管理不同项目的文档。
- **精细化上下文筛选**:
  - **三级级联筛选**: 支持按 `项目 -> 文档类型 -> 具体文件` 进行层层筛选，实现高精度问答。
  - **跨项目搜索**: 支持同时选择多个项目进行广泛的知识检索。
- **高级文档解析**: 支持 PDF, DOCX, XLSX 等多种格式，并可通过 OCR/增强模式处理扫描件和复杂表格。
- **混合检索与重排序**: 结合了向量检索（稠密）和 BM25（稀疏）的混合检索策略，并通过 Reranker 模型优化相关性排序，提升回答质量。
- **智能快捷功能**: 提供“智能摘要”、“合规风控”、“数据提取”、“深度解读”等一键式分析工具。
- **Agent 智能体 (可选)**: 集成了文书写作 Agent、Pandas Agent 等，能够执行更复杂的任务。

## 🛠️ 技术栈

- **前端框架**: [Streamlit](https://streamlit.io/)
- **核心 RAG 框架**: [LangChain](https://www.langchain.com/)
- **大语言模型 (LLM)**: [DeepSeek](https://www.deepseek.com/) (可替换)
- **向量数据库**: [ChromaDB](https://www.trychroma.com/) (可替换为 FAISS 等)
- **Embedding/Reranker 模型**: Sentence-Transformers

## 🚀 快速开始 (本地运行)

### 1. 环境准备

- 安装 [Python 3.9+](https://www.python.org/downloads/)
- (推荐) 创建并激活一个虚拟环境：
  ```bash
  python -m venv venv
  # Windows
  .\venv\Scripts\activate
  # macOS/Linux
  source venv/bin/activate
  ```

### 2. 安装依赖

将项目根目录下的 `requirements.txt` 文件中的所有依赖项进行安装：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### 3. 配置环境变量

在项目根目录创建一个名为 `.env` 的文件，并填入您的 API Key：

```
DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

*注意: `.env` 文件已被添加到 `.gitignore` 中，以防意外泄露您的密钥。*

### 4. 运行应用

执行以下命令启动 Streamlit 服务：

```bash
streamlit run app.py
```

服务启动后，在浏览器中打开显示的 URL (通常是 `http://localhost:8501`) 即可开始使用。

## 部署

本项目可以部署在任何支持 Python 的服务器上。推荐使用 Supervisor 进行进程管理，并使用 Nginx 作为反向代理以提高安全性和性能。详细步骤请参考部署指南。

## 📂 项目结构

```
.
├── data_repository/      # 存放上传的原始文件和元数据
│   ├── metadata_registry.json
│   └── ... (项目文件夹)
├── vector_db/            # 存放 ChromaDB 的向量数据
├── app.py                # Streamlit 主应用文件
├── rag_service.py        # RAG 核心服务逻辑
├── utils/
│   ├── file_manager.py   # 文件和元数据管理
│   └── etl_pipeline.py   # ETL（文档处理、向量化）流程
├── requirements.txt      # Python 依赖
├── README.md             # 本文档
└── .env                  # (本地) 环境变量文件
```
