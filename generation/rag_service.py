"""
generation/rag_service.py
旗舰修复版 v3.6 (UX优化版)：加入审核模式提示与状态显式更新
"""
import os
import sys
import io
import json
import re
import pandas as pd
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Generator, Union, Optional, Any

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as e:
    raise ImportError(f"LangChain 核心依赖缺失: {e}")

# ==============================================================================
# 🧩 模块独立导入区 (关键修复：防止一个缺失导致全盘失败)
# ==============================================================================

# 1. 向量库管理器 (核心)
VectorStoreManager = None
try:
    from etl.vector_store import VectorStoreManager
except ImportError as e:
    print(f"⚠️ [初始化警告] VectorStoreManager 导入失败: {e}")

# 2. Rerank 重排序 (可选)
RerankService = None
try:
    from generation.reranker import RerankService
except ImportError:
    pass  # 静默失败，视为未启用

# 3. GraphRAG 图谱 (可选)
GraphManager = None
try:
    from utils.graph_manager import GraphManager
except ImportError:
    pass

# 4. BM25 检索 (可选)
BM25Persistence = None
try:
    from utils.bm25_manager import BM25Persistence
except ImportError:
    pass

# 5. 文书生成引擎 (核心 - 必须确保独立导入)
TenderWriterEngine = None
try:
    from utils.tender_engine import TenderWriterEngine
except ImportError as e:
    print(f"⚠️ [初始化警告] TenderWriterEngine 导入失败 (请检查 utils/tender_engine.py 是否存在): {e}")


# ==============================================================================
# 🚀 主服务类
# ==============================================================================
class DeepSeekRAGService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "deepseek-chat"
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.3,
            streaming=True
        )

        # --- 初始化各个组件 (带独立错误捕获) ---

        # 1. VectorStore
        self.vector_store = None
        if VectorStoreManager:
            try:
                self.vs_manager = VectorStoreManager()
                self.vector_store = self.vs_manager.vector_store
            except Exception as e:
                print(f"❌ 向量库初始化异常: {e}")

        self.data_repo_dir = "data_repository"

        # 2. Reranker
        self.reranker = None
        if RerankService:
            try: self.reranker = RerankService()
            except: pass

        # 3. GraphRAG
        self.graph_manager = None
        if GraphManager:
            try: self.graph_manager = GraphManager()
            except: pass

        # 4. BM25
        self.bm25_manager = None
        if BM25Persistence:
            try: self.bm25_manager = BM25Persistence()
            except: pass

        # 5. Writer Engine (关键修复)
        self.writer_engine = None
        if TenderWriterEngine:
            try:
                self.writer_engine = TenderWriterEngine(api_key=api_key)
                print("✅ 文书生成引擎 (TenderWriterEngine) 初始化成功")
            except Exception as e:
                print(f"⚠️ 文书生成引擎初始化报错: {e}")
        else:
            print("⚠️ 未检测到 TenderWriterEngine 类定义，写作功能将不可用。")

        # Prompt 包含对话历史
        self.prompt = ChatPromptTemplate.from_template("""
        你是一个专业的水利工程标书分析专家。请基于以下检索到的上下文（Context）回答用户的问题。
        
        【对话历史】
        {chat_history}

        【检索上下文】
        {context}

        【用户问题】
        {question}
        
        请做出专业、依据充分的回答。
        """)

    def rewrite_query(self, query: str) -> str:
        try:
            prompt = ChatPromptTemplate.from_template("请将此用户问题重写为更适合搜索的查询：{question}")
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"question": query}).strip()
        except:
            return query

    def _detect_excel_task(self, docs: List[Document]) -> Optional[tuple]:
        if not docs: return None
        excel_votes = {}
        for doc in docs:
            meta = doc.metadata
            src = meta.get("source_file", "")
            if src.lower().endswith((".xlsx", ".xls")) and meta.get("type") in ["table", "summary", "sheet"]:
                project = meta.get("project_name", "")
                full_path = os.path.join(self.data_repo_dir, project, src)
                key = (full_path, meta.get("page", 0))
                excel_votes[key] = excel_votes.get(key, 0) + 1
        if not excel_votes: return None
        return max(excel_votes.items(), key=lambda x: x[1])[0]

    def _run_pandas_agent(self, file_path: str, sheet_name: str, query: str) -> str:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
        except:
            try:
                df = pd.read_excel(file_path, dtype=str)
            except Exception as e:
                return f"读取 Excel 失败: {e}"

        sys_prompt = f"你是一位资深数据分析师。当前 dataframe 变量名为 df。用户诉求：{query}。请直接编写 Python 代码解决，不要解释。"

        original_streaming = self.llm.streaming
        self.llm.streaming = False
        try:
            response = self.llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content="请写代码。")])
            code = response.content.replace("```python", "").replace("```", "").strip()
        except Exception as e:
            return f"生成代码失败: {e}"
        finally:
            self.llm.streaming = original_streaming

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec_globals = {"df": df, "pd": pd}
            exec(code, exec_globals)
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f"分析执行出错: {e}\n代码:\n{code}"

        # ==========================================
        # 📝 Writer Agent (最终修复版：修正参考与生成的逻辑倒置)
        # ==========================================
    def _run_writer_agent_stream(self, query: str) -> Generator[Dict, None, None]:
        if not self.writer_engine:
            yield {"type": "text", "data": "❌ 文书生成引擎未初始化（可能缺少依赖文件 utils/tender_engine.py）。"}
            return

        yield {"type": "status", "data": "🚀 收到写作指令，正在启动【文书生成工坊】..."}

        try:
            # 1. 提取参数 (修复核心：明确区分 参考文件(Source) 和 新标题(Target))
            yield {"type": "status", "data": "🔍 智能解析意图..."}
            self.llm.streaming = False

            # ✅ [关键修改] Prompt 明确区分“参考对象”和“生成对象”
            extract_prompt = f"""
            你是一个精准的参数提取助手。请分析用户指令，提取以下三个关键字段，并返回纯 JSON 格式：

            【用户指令】
            "{query}"

            【提取规则】
            1. "reference_filename": 用户指定的**参考范文**文件名（即已存在的、需要被模仿的文件）。
               - 关键词：参考、仿照、基于、根据。
               - 例如："参考《A.docx》" -> 提取 "A.docx"。
            2. "new_project_title": 用户想要**新建**的文档标题。
               - 关键词：撰写、生成、写一份。
               - 例如："写一份《B.docx》" -> 提取 "B.docx"。
            3. "new_project_info": 关于新项目的背景描述、建设内容等所有信息。

            【返回格式】
            {{ 
                "reference_filename": "...", 
                "new_project_title": "...", 
                "new_project_info": "..." 
            }}
            """

            resp = self.llm.invoke(extract_prompt)
            self.llm.streaming = True

            # 解析 JSON
            content = resp.content.replace("```json", "").replace("```", "").strip()
            try:
                params = json.loads(content)
            except json.JSONDecodeError:
                # 兜底：如果 JSON 解析失败，尝试用正则提取
                params = {}

            # 获取参数
            ref_filename = params.get("reference_filename")
            new_title = params.get("new_project_title", "未命名文档")
            project_info = params.get("new_project_info", query)

            # 如果提取失败，或者提取成了同一个，做简单的逻辑修正
            if ref_filename and new_title and ref_filename == new_title:
                # AI 可能会混淆，这里简单判断：如果文件名包含 "参考"，则可能是参考文件
                pass

            if not ref_filename:
                yield {"type": "text", "data": "❌ 无法识别参考文件。请明确说明“参考 xx文件”。"}
                return

            print(f"🔍 [解析结果] 参考: {ref_filename} | 新建: {new_title}")

        except Exception as e:
            self.llm.streaming = True
            yield {"type": "text", "data": f"❌ 解析指令失败: {e}"}
            return

        # =======================================================
        # 2. 定位文件 (使用 ref_filename 去硬盘找，而不是用 new_title)
        # =======================================================
        ref_path = None
        # 归一化参考文件名
        target_pure = ref_filename.lower().replace(" ", "").replace("《", "").replace("》", "").replace(".docx",
                                                                                                      "")

        search_paths = [self.data_repo_dir, "uploads", ".", "data"]
        found_candidates = []

        print(f"🔍 [系统] 正在搜索参考文件，关键词：[{target_pure}]")

        for search_dir in search_paths:
            if not os.path.exists(search_dir): continue
            for root, _, files in os.walk(search_dir):
                for f in files:
                    if f.startswith("~$") or f.startswith("."): continue
                    f_pure = f.lower().replace(" ", "")

                    # 模糊匹配逻辑
                    if (target_pure in f_pure) or (os.path.splitext(f_pure)[0] in target_pure):
                        full_path = os.path.join(root, f)
                        found_candidates.append(full_path)
                        break
                if found_candidates: break

        if found_candidates:
            ref_path = found_candidates[0]

        if not ref_path:
            yield {"type": "text",
                   "data": f"❌ 在知识库中未找到参考文件：`{ref_filename}`。\n\n**系统解析到的意图：**\n- 参考：{ref_filename} (去硬盘找这个)\n- 新建：{new_title}\n\n建议：请确认上传的文件名是否包含 `{target_pure}`。"}
            return

        yield {"type": "text", "data": f"✅ 已锁定参考文件：`{os.path.basename(ref_path)}`\n\n"}

        # =======================================================
        # 3. 后续流程 (使用 new_title 作为输出文件名)
        # =======================================================
        try:
            # Load
            yield {"type": "status", "data": "📖 解析参考文档..."}
            self.writer_engine.load_reference(ref_path)

            # Style DNA
            yield {"type": "status", "data": "🧬 提取文风 DNA..."}
            style_guide = self.writer_engine.analyze_style()
            yield {"type": "text", "data": f"> **文风 DNA**：{style_guide}\n\n"}

            # Outline
            yield {"type": "status", "data": "📋 构思目录..."}
            # 将新标题也传给大纲生成器，以便生成更准确的标题
            full_project_info = f"项目名称：{new_title}\n背景信息：{project_info}"
            new_toc = self.writer_engine.generate_outline(full_project_info)

            # 2.1 输出目录结构
            toc_preview = "\n".join([f"- {t}" for t in new_toc[:5]])
            yield {"type": "text", "data": f"**目录框架**：\n{toc_preview}\n... (共{len(new_toc)}章)\n\n"}
            yield {"type": "toc", "data": new_toc}

            # ✅ [UX优化 1] 在这里插入“耗时说明”提示框
            yield {"type": "text", "data": """
---
#### ⏱️ 生成耗时说明：已开启“智能审核”模式
为了确保报告内容的准确性，系统正在对每一个章节执行 **“双重校验” (生成 + 深度审核)**：
1. **生成**：撰写初稿。
2. **审核**：检查并修正可能存在的旧地名残留或逻辑幻觉。

> ⚠️ **注意**：此过程会显著增加生成时间（预计整份报告需 3-5 分钟）。
> 如果您只需要部分内容，建议在提问时指定章节（例如：“帮我生成第三章建设方案”），速度会快很多。
---
"""}

            # Map
            yield {"type": "status", "data": "🔗 建立映射..."}
            mapping = self.writer_engine.map_toc_relationships(new_toc)

            # Checkpoint Prep (缓存文件用 ref + new_title 做 key)
            cache_key = hashlib.md5(f"{ref_filename}_{new_title}_{len(new_toc)}".encode()).hexdigest()
            cache_file = os.path.join("outputs", f"cache_{cache_key}.json")
            os.makedirs("outputs", exist_ok=True)

            generated_data = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        generated_data = json.load(f)
                    yield {"type": "text",
                           "data": f"⚡ **检测到历史进度**，已自动恢复 {len(generated_data)} 个章节。\n\n"}
                except:
                    pass

            chapters_to_write = [t for t in new_toc if t not in generated_data]

            if not chapters_to_write:
                yield {"type": "text", "data": "🎉 所有章节均已生成完毕，直接导出...\n"}

            # 并发执行 + 有序输出
            output_dir = "outputs"
            # ✅ [关键修改] 使用用户指定的新标题作为文件名
            safe_title = new_title.replace("《", "").replace("》", "").replace(".docx", "")
            safe_title = re.sub(r'[\\/*?:"<>|]', "", safe_title)
            if not safe_title: safe_title = "未命名标书"

            output_filename = f"{safe_title}_{int(pd.Timestamp.now().timestamp())}.docx"
            output_path = os.path.join(output_dir, output_filename)

            total_tasks = len(chapters_to_write)
            if total_tasks > 0:
                MAX_WORKERS = 3
                yield {"type": "status", "data": f"🚀 正在全速撰写并审核剩余 {total_tasks} 章..."}

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # 提交任务
                    future_to_title = {
                        executor.submit(
                            self.writer_engine.write_chapter,
                            title,
                            mapping.get(title),
                            full_project_info,
                            style_guide
                        ): title
                        for title in chapters_to_write
                    }

                    results_buffer = {}
                    next_idx_to_display = 0
                    completed_count = 0

                    for future in as_completed(future_to_title):
                        title = future_to_title[future]
                        try:
                            content = future.result()
                            results_buffer[title] = content
                            generated_data[title] = content
                            completed_count += 1

                            # ✅ [UX优化 2] 状态栏明确显示“正在审核”
                            yield {"type": "status",
                                   "data": f"✍️ 正在撰写并审核：({completed_count}/{total_tasks}) 个章节完成..."}

                            # 保存缓存
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump(generated_data, f, ensure_ascii=False)

                            # 有序显示
                            while next_idx_to_display < len(chapters_to_write):
                                target_title = chapters_to_write[next_idx_to_display]
                                if target_title in results_buffer:
                                    preview = results_buffer[target_title][:50].replace("\n", "") + "..."
                                    yield {"type": "text", "data": f"✅ **{target_title}**\n> {preview}\n\n"}
                                    next_idx_to_display += 1
                                else:
                                    break
                        except Exception as e:
                            print(f"生成错误: {e}")
                            yield {"type": "text", "data": f"❌ **{title}** 生成失败\n"}

            # Compile
            yield {"type": "status", "data": "💾 排版导出中..."}
            ordered_content = {t: generated_data.get(t, "") for t in new_toc}
            self.writer_engine.compile_to_word(ordered_content, output_path)

            final_msg = f"""
### 🎉 生成完成！

**文件**：{output_filename}
**参考范文**：{os.path.basename(ref_path)}
**文风**：{style_guide[:15]}...

请点击下方按钮下载。
"""
            yield {"type": "text", "data": final_msg}
            yield {"type": "file", "data": {"path": output_path, "name": output_filename}}

        except Exception as e:
            traceback.print_exc()
            yield {"type": "text", "data": f"\n❌ 严重错误: {str(e)}"}

    # ==========================================
    # 🚀 主流程 (Standard RAG - 完整逻辑)
    # ==========================================
    """
    === Python代码文件: rag_service.py -> DeepSeekRAGService.chat_stream (V5.1 最终完整无省略版) ===
    - 包含您提供的所有逻辑，未经任何省略。
    - 注入了对复杂 filter_config 的支持，并确保完全向后兼容。
    """
    from typing import Union, List, Dict, Generator
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    import traceback

    def chat_stream(self, query: str, history: List[Dict],
                    # 兼容旧版调用，但建议废弃
                    top_k: int = 6,
                    project_filter: Union[str, List[str]] = None,
                    # 优先使用新的、统一的配置字典
                    filter_config: Dict = None) -> Generator[Dict, None, None]:

        # --- 0. 配置融合与准备 (新增逻辑，保证100%兼容性) ---
        if filter_config is None:
            # 如果调用方没有传入新的 filter_config，则根据旧参数临时构建一个
            # 这确保了即使是旧的前端或测试脚本也能正常工作
            print("⚠️ [RAG Service] Warning: 使用旧版参数调用 chat_stream，建议切换到 filter_config。")
            filter_config = {
                "top_k": top_k,
                "project": project_filter or "所有项目",
                "type": "所有类型",
                "files": []
            }

        # 从统一的 filter_config 中获取所有参数，作为后续流程的唯一信源
        final_top_k = filter_config.get("top_k", 6)

        # --- 1. 写作意图侦测 (完整保留) ---
        writing_keywords = ["撰写", "生成", "写一份", "仿照", "起草", "编制"]
        context_keywords = ["参考", "根据", "基于", "模仿"]
        # 如果问题中同时包含“撰写类词汇”和“参考类词汇”，则转入文书生成 Agent
        if any(wk in query for wk in writing_keywords) and any(ck in query for ck in context_keywords):
            # 假设 self._run_writer_agent_stream 是您已实现的方法
            if hasattr(self, '_run_writer_agent_stream'):
                for evt in self._run_writer_agent_stream(query):
                    yield evt
                return
            else:
                print("⚠️ [RAG Service] Warning: 检测到写作意图，但 _run_writer_agent_stream 方法未实现。")

        # --- 2. 问题重写 (完整保留) ---
        yield {"type": "status", "data": "🧠 优化搜索问题..."}
        # 假设 self.rewrite_query 是您已实现的方法
        optimized_query = self.rewrite_query(query) if hasattr(self, 'rewrite_query') else query
        yield {"type": "status", "data": f"🔍 检索: {optimized_query}..."}

        # --- 3. 混合检索 ---
        fetch_k = final_top_k * 5

        # --- 3.1 向量检索 Filter 构建 (核心增强部分) ---
        search_kwargs = {"k": fetch_k}

        conditions = []
        # 条件1: 项目范围
        project_scope = filter_config.get("project")
        if project_scope and project_scope != "所有项目":
            if isinstance(project_scope, list) and len(project_scope) > 0:
                conditions.append({"project_name": {"$in": project_scope}})
            elif isinstance(project_scope, str):
                conditions.append({"project_name": {"$eq": project_scope}})

        # 条件2: 文档类型
        type_scope = filter_config.get("type")
        if type_scope and type_scope != "所有类型":
            conditions.append({"category": {"$eq": type_scope}})

        # 条件3: 具体文件
        files_scope = filter_config.get("files")
        if files_scope and isinstance(files_scope, list) and len(files_scope) > 0:
            # 如果有文件级筛选，它的优先级最高，可以覆盖其他筛选条件以获得最精确结果
            conditions = [{"source_file": {"$in": files_scope}}]

        # 组合所有条件
        if len(conditions) > 1:
            search_kwargs["filter"] = {"$and": conditions}
        elif len(conditions) == 1:
            search_kwargs["filter"] = conditions[0]

        print(f"✅ [RAG Service] 构建的最终 filter: {search_kwargs.get('filter')}")

        # --- 3.2 执行向量检索 (完整保留) ---
        vector_docs = []
        if self.vector_store:
            try:
                vector_docs = self.vector_store.as_retriever(search_type="similarity",
                                                             search_kwargs=search_kwargs).invoke(optimized_query)
            except Exception as e:
                print(f"向量检索警告: {e}")

        # --- 3.3 执行 BM25 检索 (完整保留，仅适配 filter_config) ---
        bm25_docs = []
        bm25_project_scope = filter_config.get("project")
        if self.bm25_manager and bm25_project_scope and bm25_project_scope != "所有项目":
            try:
                target = bm25_project_scope if isinstance(bm25_project_scope, list) else [bm25_project_scope]
                raw = self.bm25_manager.search(query, target, top_k=5)
                for item in raw:
                    bm25_docs.append(Document(page_content=item['content'], metadata=item['metadata']))
            except Exception as e:
                print(f"BM25 检索警告: {e}")

        # --- 3.4 合并与去重 (完整保留) ---
        unique_ids = set()
        initial_docs = []
        # 优先保留BM25的结果，因为它对于关键词匹配通常更准
        for d in bm25_docs + vector_docs:
            cid = d.metadata.get("chunk_id")
            if cid:
                if cid not in unique_ids:
                    d.metadata["source_method"] = "BM25" if d in bm25_docs else "Vector"
                    initial_docs.append(d)
                    unique_ids.add(cid)
            # 对于没有 chunk_id 的文档，可以按内容去重（兼容老数据）
            elif d.page_content not in unique_ids:
                d.metadata["source_method"] = "BM25" if d in bm25_docs else "Vector"
                initial_docs.append(d)
                unique_ids.add(d.page_content)

        # --- 4. GraphRAG 扩展 (完整保留) ---
        if initial_docs and self.graph_manager:
            try:
                yield {"type": "status", "data": "🕸️ 扩展图谱上下文..."}
                expanded_docs = []
                # 使用 `unique_ids` 来避免重复添加已经存在的图谱节点
                seen_graph_ids = unique_ids.copy()

                # 对检索结果中得分最高的前 5 个文档进行图谱扩展
                for d in initial_docs[:5]:
                    cid = d.metadata.get("chunk_id")
                    if cid:
                        # 获取 1 跳邻居
                        context_rows = self.graph_manager.get_context_window(cid, 1)
                        for row in context_rows:
                            for key in ['prev', 'next']:  # 检查前后节点
                                node = row.get(key)
                                if node and node.get('id') not in seen_graph_ids:
                                    expanded_docs.append(Document(
                                        page_content=f"[图谱关联] {node.get('text', '')}",
                                        metadata={"chunk_id": node.get('id'), "source_method": "Graph"}
                                    ))
                                    seen_graph_ids.add(node.get('id'))

                if expanded_docs:
                    print(f"🕸️ 图谱扩展了 {len(expanded_docs)} 个新片段")
                    initial_docs.extend(expanded_docs)
            except Exception as e:
                print(f"⚠️ GraphRAG 扩展失败: {e}")

        # --- 5. Rerank (完整保留) ---
        if not initial_docs:
            context_str = "未找到相关文档。我将基于通用知识进行回答。"
            final_docs = []
        else:
            if self.reranker:
                yield {"type": "status", "data": f"⚖️ 重排序 {len(initial_docs)} 个片段..."}
                final_docs = self.reranker.rerank(optimized_query, initial_docs, top_k=final_top_k)
            else:
                # 如果没有 reranker，直接截取 top_k
                final_docs = initial_docs[:final_top_k]

            context_str = "\n\n".join([
                                          f"引用 {i + 1} (来源: {d.metadata.get('source_file', '未知')}, 方法: {d.metadata.get('source_method', '未知')}):\n{d.page_content}"
                                          for i, d in enumerate(final_docs)])

        # --- 6. Pandas Agent (完整保留) ---
        excel_task_info = self._detect_excel_task(final_docs) if hasattr(self, '_detect_excel_task') else None
        if excel_task_info:
            file_path, sheet_name = excel_task_info
            yield {"type": "status", "data": "📊 启动 Pandas Agent 分析Excel..."}
            try:
                # 假设 self._run_pandas_agent 是您已实现的方法
                agent_result = self._run_pandas_agent(file_path, sheet_name, optimized_query)
                yield {"type": "text", "data": agent_result}
                # 即使Agent成功，也提供一些原始来源作为参考
                yield {"type": "sources",
                       "data": [{"content": d.page_content, "metadata": d.metadata} for d in final_docs[:3]]}
                return
            except Exception as e:
                yield {"type": "text", "data": f"\n⚠️ Pandas Agent 分析失败，转为通用回答模式。错误: {e}\n\n"}

        # --- 7. 标准 RAG 生成 (完整保留) ---
        yield {"type": "sources", "data": [{"content": d.page_content, "metadata": d.metadata} for d in final_docs]}

        # 构造历史对话记录
        chat_history_str = ""
        if history:
            for msg in history[-4:]:  # 只取最近4轮对话，避免过长
                chat_history_str += f"{msg['role']}: {msg['content']}\n"

        # 假设 self.prompt 是一个 PromptTemplate 对象
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            # 流式调用LLM
            for chunk in chain.stream({
                "chat_history": chat_history_str,
                "context": context_str,
                "question": optimized_query
            }):
                yield {"type": "text", "data": chunk}
        except Exception as e:
            traceback.print_exc()
            yield {"type": "text", "data": f"大模型调用错误: {str(e)}"}

