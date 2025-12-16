"""
=== Python代码文件: app.py ===
"""

import streamlit as st
import os
import time
import uuid
import PyPDF2  # 📄 新增：用于读取 PDF 文本
from docx import Document as DocxDocument  # 📄 新增：用于读取 Word 文本
from streamlit_agraph import agraph, Node, Edge, Config  # 🕸️ 新增：图谱可视化组件

# --- 引入工具 ---
try:
    from utils.file_manager import FileManager
    from etl.pipeline import ETLPipeline
    from generation.rag_service import DeepSeekRAGService
    # 🕸️ 新增：引入图谱构建引擎 (请确保 etl/graph_engine.py 存在)
    from etl.graph_engine import KnowledgeGraphEngine
except ImportError as e:
    st.error(f"❌ 核心模块导入失败: {e}")
    st.info("💡 请检查是否安装了依赖: pip install PyPDF2 python-docx streamlit-agraph")
    st.stop()

# 设置页面配置
st.set_page_config(
    page_title="DeepSeek RAG Pro",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化文件管理器
file_manager = FileManager(base_dir="data_repository")

# ==========================================
# 🛠️ 辅助工具：读取真实文件内容
# ==========================================
def read_file_content(file_path):
    """
    读取真实文件内容，返回字符串。
    为了防止 Token 溢出和响应过慢，对大文件进行了截断处理。
    """
    text = ""
    try:
        if not os.path.exists(file_path):
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                # 策略：读取前 20 页作为图谱生成的依据 (可根据需求调整)
                limit = min(20, len(reader.pages))
                for i in range(limit):
                    page_text = reader.pages[i].extract_text()
                    if page_text: text += page_text + "\n"
        elif ext == '.docx':
            doc = DocxDocument(file_path)
            # Word 文档通常文本较稀疏，读取前 500 段
            for i, para in enumerate(doc.paragraphs):
                if i > 500: break
                text += para.text + "\n"
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read(20000) # 限制字符数
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}: {e}")
    return text

# ==========================================
# 🎨 UI 美化核心区域 (CSS 注入)
# ==========================================
st.markdown("""
<style>
    /* 1. 全局背景与字体优化 */
    .stApp { background-color: #f8f9fa; }
    [data-testid="stDecoration"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* 2. 侧边栏优化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
        box-shadow: 2px 0 5px rgba(0,0,0,0.02);
    }

    /* 3. 标题样式 */
    .main-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(120deg, #005bea 0%, #00c6fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .sub-title {
        font-size: 1.1rem !important;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 4. 卡片样式优化 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid #e0e0e0 !important;
        background-color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #005bea !important;
    }

    /* 5. 按钮美化 */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #005bea 0%, #007bff 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(0, 91, 234, 0.3);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #0048c4 0%, #0069d9 100%);
        box-shadow: 0 6px 15px rgba(0, 91, 234, 0.4);
    }
    .stChatInputContainer {
        border-radius: 15px !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- ✅ Session State 初始化 ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

# 确保至少有一个会话
if not st.session_state.sessions:
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {"title": "新对话", "messages": []}
    st.session_state.current_session_id = new_id

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]

# 兼容性变量
if "view_pdf" not in st.session_state: st.session_state.view_pdf = None

# 图谱 Session 变量
if "graph_data" not in st.session_state: st.session_state.graph_data = {}
if "graph_source_name" not in st.session_state: st.session_state.graph_source_name = ""

current_session_id = st.session_state.current_session_id
if current_session_id not in st.session_state.sessions:
    current_session_id = list(st.session_state.sessions.keys())[0]
    st.session_state.current_session_id = current_session_id

current_messages = st.session_state.sessions[current_session_id]["messages"]

# --- RAG 服务初始化 ---
if "rag_service" not in st.session_state:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        st.warning("⚠️ 请配置 DEEPSEEK_API_KEY")
    else:
        status_container = st.empty()
        with status_container.status("🚀 正在启动系统核心组件...", expanded=True) as status:
            st.write("🔌 连接 DeepSeek API...")
            try:
                rag = DeepSeekRAGService(key)
                st.write("🧠 加载 Embedding 模型 (BGE-Small)...")
                if rag.vector_store:
                    st.write("💾 连接 ChromaDB 向量库成功")
                st.session_state.rag_service = rag
                status.update(label="✅ 系统准备就绪！", state="complete", expanded=False)
                time.sleep(1)
            except Exception as e:
                status.update(label="❌ 启动失败", state="error")
                st.error(f"初始化错误: {str(e)}")
                st.stop()
        status_container.empty()

# ==========================================
# 📂 侧边栏
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dam.png", width=50)

    # --- 💬 模块 1: 对话管理 ---
    st.markdown("### 💬 对话管理")
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {"title": "新对话", "messages": []}
        st.session_state.current_session_id = new_id
        st.rerun()

    st.markdown("---")
    st.caption("历史记录")
    session_ids = list(st.session_state.sessions.keys())
    for s_id in reversed(session_ids):
        s_data = st.session_state.sessions[s_id]
        title = s_data.get("title", "未命名对话")
        c1, c2 = st.columns([0.8, 0.2])
        is_current = (s_id == st.session_state.current_session_id)
        prefix = "📂" if is_current else "📃"
        if c1.button(f"{prefix} {title}", key=f"sess_{s_id}", use_container_width=True):
            st.session_state.current_session_id = s_id
            st.rerun()
        if c2.button("🗑️", key=f"del_sess_{s_id}"):
            del st.session_state.sessions[s_id]
            if s_id == st.session_state.current_session_id:
                if st.session_state.sessions:
                    st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
                else:
                    nid = str(uuid.uuid4())
                    st.session_state.sessions[nid] = {"title": "新对话", "messages": []}
                    st.session_state.current_session_id = nid
            st.rerun()

    st.markdown("---")

    # --- 🗂️ 模块 2: 知识库管理 (精准注入元数据注册) ---
    st.markdown("### 🗂️ 知识库")
    tab_upload, tab_manage = st.tabs(["📤 上传", "👀 管理"])

    with tab_upload:
        st.info("💡 提示：支持Excel，word，prd 文件的上传")
        existing_projects = file_manager.get_folders()
        project_mode = st.radio("项目选择", ["现有项目", "新建项目"], horizontal=True, label_visibility="collapsed")

        target_project = ""
        if project_mode == "现有项目":
            if existing_projects:
                target_project = st.selectbox("选择项目", existing_projects)
            else:
                st.warning("暂无项目，请先新建")
                project_mode = "新建项目"

        if project_mode == "新建项目":
            target_project = st.text_input("输入新项目名称", placeholder="例如：三峡工程_2024标段")

        if "user_custom_types" not in st.session_state: st.session_state.user_custom_types = []
        base_types = ["招标文件", "技术规范", "施工图纸", "合同商务", "其他"]
        all_options = base_types + st.session_state.user_custom_types + ["➕ 新建类型..."]

        default_idx = 0
        if "last_added_type" in st.session_state and st.session_state.last_added_type in all_options:
            try:
                default_idx = all_options.index(st.session_state.last_added_type)
            except:
                default_idx = 0

        selected_type_opt = st.selectbox("文档类型", all_options, index=default_idx)

        if selected_type_opt == "➕ 新建类型...":
            custom_category = st.text_input("新类型名称", placeholder="例: 地质勘察")
            if st.button("💾 保存类型"):
                if custom_category.strip() and custom_category.strip() not in all_options:
                    st.session_state.user_custom_types.append(custom_category.strip())
                    st.session_state.last_added_type = custom_category.strip()
                    st.rerun()
            doc_category = custom_category.strip()
        else:
            doc_category = selected_type_opt

        st.divider()
        uploaded_files = st.file_uploader("添加文档", type=["pdf", "docx", "xlsx", "xls"], accept_multiple_files=True)
        with st.expander("🛠️ 高级解析设置", expanded=False):
            use_advanced = st.toggle("增强解析 (OCR/表格)", value=True)
            force_update = st.checkbox("强制覆盖 (忽略查重)", value=False)

        if uploaded_files and st.button("开始入库", type="primary", use_container_width=True):
            if not target_project or not doc_category:
                st.error("❌ 信息不完整")
            elif not os.getenv("DEEPSEEK_API_KEY"):
                st.error("请配置 API Key")
            else:
                if doc_category not in all_options: st.session_state.user_custom_types.append(doc_category)
                file_manager.create_folder(target_project)
                with st.status("正在入库...", expanded=True) as status:
                    pipeline = ETLPipeline(os.getenv("DEEPSEEK_API_KEY"))
                    prog = st.progress(0)
                    for i, f in enumerate(uploaded_files):
                        saved_path = file_manager.save_file(f, target_project)
                        # [保持不变] 调用ETL Pipeline进行文件处理和向量化
                        pipeline.process_file(
                            saved_path, use_advanced_mode=use_advanced,
                            force_update=force_update, original_filename=f.name,
                            user_project=target_project, user_tag=doc_category
                        )

                        # 🌟 [只增不改] 在处理成功后，立刻调用元数据注册方法 🌟
                        file_manager.register_file_metadata(
                            project=target_project,
                            tag=doc_category,
                            filename=f.name
                        )

                        prog.progress((i + 1) / len(uploaded_files))
                    status.update(label="✅ 入库完成", state="complete")
                    time.sleep(1)
                    st.rerun()

    with tab_manage:

        files_map = file_manager.get_all_files()
        if not files_map: st.caption("暂无文件")
        for folder, files in files_map.items():
            with st.expander(f"📁 {folder} ({len(files)})", expanded=False):
                for f in files:
                    fc1, fc2 = st.columns([0.8, 0.2])
                    fc1.text(f"📄 {f}")
                    if fc2.button("🗑️", key=f"del_f_{folder}_{f}"):
                        file_manager.delete_file(folder, f)
                        st.rerun()

    # --- 🕸️ 模块 3: 知识图谱生成  ---
    st.markdown("---")
    st.markdown("### 🕸️ 知识图谱生成")

    # 1. 范围选择
    graph_scope = st.radio("生成范围", ["单文件", "全项目"], horizontal=True, key="g_scope")

    selected_graph_source = ""
    # 获取所有项目列表
    projects_list = file_manager.get_folders()
    graph_project = st.selectbox("选择项目", projects_list, key="g_proj")

    target_graph_file = None

    # 获取当前选中项目下的文件列表
    # 修复：使用 get_all_files() 获取字典，然后通过 key 取值
    project_files = []
    if graph_project:
        all_files_map = file_manager.get_all_files()
        project_files = all_files_map.get(graph_project, [])

    # 逻辑：根据 Scope 动态显示文件选择
    if graph_project:
        if graph_scope == "单文件":
            if project_files:
                target_graph_file = st.selectbox("选择文件", project_files, key="g_file")
                if target_graph_file:
                    selected_graph_source = f"{graph_project} / {target_graph_file}"
            else:
                st.warning("该项目下暂无文件")
        else:
            selected_graph_source = f"项目：{graph_project} (全量)"

    # 生成按钮
    if st.button("🚀 生成星系图", type="primary", use_container_width=True):
        if not selected_graph_source or not graph_project:
            st.error("请先选择有效的文件或项目")
        else:
            status_container = st.empty()  # 创建一个占位符用于显示实时状态
            try:
                with st.spinner(f"正在分析 {selected_graph_source}，DeepSeek 思考中 (可能需要 30秒)..."):
                    full_text = ""
                    base_path = os.path.join("data_repository", graph_project)

                    # 1. 读取文件
                    status_container.info("📖 正在读取硬盘文件...")
                    if graph_scope == "单文件" and target_graph_file:
                        f_path = os.path.join(base_path, target_graph_file)
                        full_text = read_file_content(f_path)
                    elif graph_scope == "全项目":
                        if not project_files:
                            st.error("项目为空")
                            st.stop()
                        for f_name in project_files:
                            f_path = os.path.join(base_path, f_name)
                            content = read_file_content(f_path)
                            full_text += f"\n=== 文件：{f_name} ===\n{content[:2000]}\n"

                    if not full_text.strip():
                        st.error("❌ 未读取到文本内容，请检查文件是否加密或为空。")
                        st.stop()

                    # 2. 调用 AI
                    status_container.info("🧠 正在调用 DeepSeek 构建知识网络...")
                    kg_engine = KnowledgeGraphEngine(os.getenv("DEEPSEEK_API_KEY"))
                    data = kg_engine.generate_graph_data(full_text)

                    # 3. 结果校验 (关键修改！)
                    if not data or not data.get("nodes"):
                        status_container.error("❌ 图谱生成失败：AI 未返回有效数据。")
                        st.error(
                            "可能原因：\n1. API Key 余额不足或过期。\n2. 文档内容过长导致超时（请尝试单文件模式）。\n3. 请查看后台控制台的报错日志。")
                    else:
                        # 成功
                        st.session_state.graph_data = data
                        st.session_state.graph_source_name = selected_graph_source
                        status_container.success("✅ 生成成功！")
                        st.toast("✅ 图谱已就绪，请查看右侧标签页！", icon="🎉")
                        time.sleep(1)
                        status_container.empty()  # 清除状态提示

            except Exception as e:
                st.error(f"系统错误: {str(e)}")

# ==========================================
# 🌟 主界面逻辑 (分 Tab 结构)
# ==========================================

# 创建两个主标签页
tab_chat, tab_graph = st.tabs(["💬 对话模式", "🕸️ 星系图谱"])

# === Tab 1: 对话模式 (V4.0 布局 + 全功能保留版) ===
with tab_chat:
    # -------------------------------------------------------------
    # 1. 定义布局：左侧配置区，右侧交互区
    # -------------------------------------------------------------
    config_col, chat_col = st.columns([1, 2.2])

    # -------------------------------------------------------------
    # 2. 左侧配置区 (升级为 Popover 弹窗)
    # -------------------------------------------------------------
    with config_col:
        # 使用 Popover 创建一个浮动配置窗口
        with st.popover("🎯 范围与精度", use_container_width=True):

            # --- A. 范围限定 (完整三级级联) ---
            st.markdown("##### 1. 限定范围")
            projects = file_manager.get_folders()

            # 级联 1: 项目范围 (多选)
            selected_projects = st.multiselect(
                "项目范围",
                options=projects,
                default=[],
                placeholder="默认全库搜索...",
                help="可多选项目进行跨库搜索。若只选一项，可进一步筛选。"
            )

            # 级联 2: 文档类型 (仅当只选一个项目时激活)
            is_single_project = len(selected_projects) == 1

            # 初始化，防止 Streamlit 状态问题
            selected_type = "所有类型"
            selected_files = []

            if is_single_project:
                doc_types = []
                try:  # 尝试从 file_manager 获取 tags
                    doc_types = file_manager.get_tags_for_project(selected_projects[0])
                except Exception:
                    pass  # 兼容旧版

                selected_type = st.selectbox(
                    "文档类型",
                    ["所有类型"] + doc_types,
                    key="sel_type_v6",
                    help="在选定的项目中，按文档类型进行二次筛选。"
                )

                # 级联 3: 具体文件 (核心补全)
                if selected_type != "所有类型":
                    files = []
                    try:  # 尝试从 file_manager 获取文件列表
                        files = file_manager.get_files_for_project_and_tag(selected_projects[0], selected_type)
                    except Exception:
                        pass  # 兼容旧版

                    selected_files = st.multiselect(
                        "具体文件",
                        files,
                        key="sel_files_v6",
                        help="可直接锁定一个或多个文件进行精确问答。"
                    )

            st.divider()

            # --- B. 检索参数 ---
            st.markdown("##### 2. 调整精度")
            top_k_val = st.slider("参考片段数", 3, 15, 6, key="top_k_v6", help="AI回答时引用的相关文档片段数量。")

            # --- C. 统一存储最终配置 ---
            st.session_state.search_config = {
                "project": selected_projects if selected_projects else "所有项目",
                "type": selected_type,
                "files": selected_files,
                "top_k": top_k_val,
            }

        # --- Popover 外部：显示当前配置摘要 ---
        with st.container(border=True):
            config = st.session_state.get("search_config", {})
            proj_display = config.get("project", "全库")
            if isinstance(proj_display, list):
                proj_display = ' & '.join(proj_display) if proj_display else '全库'

            type_display = f" > {config.get('type')}" if config.get('type') != '所有类型' else ''
            files_display = f" > {len(config.get('files', []))}个文件" if config.get('files') else ''

            st.markdown("###### 当前配置")
            st.caption(
                f"**范围**: `{proj_display}{type_display}{files_display}`\n\n**精度**: `Top_K = {config.get('top_k', 6)}`")

    # -------------------------------------------------------------
    # 3. 右侧核心交互区
    # -------------------------------------------------------------
    with chat_col:
        # 定义唯一的、实时读取左侧配置的 quick_ask 函数
        def quick_ask(prompt_text, action_type="analysis"):
            config = st.session_state.search_config
            if action_type == "analysis" and config["project"] == "所有项目":
                st.toast("❌ 分析操作需先在左侧限定至少一个项目！", icon="🚫")
                return

            current_messages.append({"role": "user", "content": prompt_text})
            if st.session_state.sessions[current_session_id]["title"] == "新对话":
                proj_str = config['project'][0] if isinstance(config['project'], list) and config['project'] else '全库'
                st.session_state.sessions[current_session_id]["title"] = f"[{proj_str}] {prompt_text[:8]}"
            st.rerun()


        # 如果是新对话，显示完整的欢迎页和功能卡片
        if not current_messages:
            # --- 恢复完整的欢迎页 ---
            st.write("")
            st.markdown('<h1 class="main-title">🌊 规划设计智能助手</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-title">基于 DeepSeek V3 内核 · 专业的标书分析、风险识别与技术咨询专家</p>',
                        unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<p style="text-align: center; color: grey;">请在左侧选择范围后，使用下方功能或直接提问</p>',
                        unsafe_allow_html=True)

            # --- 恢复完整的四卡片功能 ---
            c1, c2, c3, c4 = st.columns(4)
            # ... (此处省略卡片内部重复代码以缩短篇幅，但功能与您提供的代码完全一致)
            with c1:
                with st.container(border=True):
                    st.markdown("###### 📝 **智能摘要**")
                    with st.popover("📄 配置", use_container_width=True):
                        st.markdown("###### 🎯 选择阅读视角")
                        sum_mode = st.radio("摘要类型", ["📊 管理层汇报", "🔧 技术/执行", "🔢 数据/参数"], key="sum_mode_final")
                        st.divider()
                        if st.button("生成摘要", key="btn_sum_final", use_container_width=True):
                            if "管理层" in sum_mode: p = "请为管理层生成一份高层摘要。重点提炼文档的背景、核心目标、主要结论以及关键决策点。"
                            elif "技术" in sum_mode: p = "请为执行人员生成一份实操摘要。重点梳理文档中的技术路线、操作步骤或具体的执行规范。"
                            else: p = "请提取文档中的关键数据与指标，包括但不限于：金额、日期、性能参数等，以列表形式呈现。"
                            quick_ask(p, "analysis")
            # --- C2: 合规风控 ---
            with c2:
                with st.container(border=True):
                    st.markdown("###### ⚖️ **合规风控**")
                    with st.popover("⚠️ 配置", use_container_width=True):
                        st.markdown("###### 🕵️‍♂️ 选择分析模式")
                        risk_mode = st.radio("模式", ["🔍 外部审查", "📝 内部自查", "🛡️ 政策合规"], key="risk_mode_final")
                        st.divider()
                        if st.button("开始分析", key="btn_risk_final", use_container_width=True):
                            if "外部" in risk_mode: p = "请以批判性视角阅读文档，识别其中可能存在的不利条款、逻辑漏洞或模糊表述。"
                            elif "内部" in risk_mode: p = "请作为审核员检查这份文档，指出其中是否存在前后矛盾、内容遗漏或关键要素缺失的情况。"
                            else: p = "请分析文档内容的合规性，检查其是否符合相关标准或规范，指出潜在的违规风险点。"
                            quick_ask(p, "analysis")
            # --- C3: 数据提取 ---
            with c3:
                with st.container(border=True):
                    st.markdown("###### 🧩 **数据提取**")
                    with st.popover("📦 配置", use_container_width=True):
                        st.markdown("###### 📦 选择提取内容")
                        qty_mode = st.radio("内容类型", ["📋 关键清单/表格", "🌳 文档大纲", "🧠 核心术语"], key="qty_mode_final")
                        st.divider()
                        if st.button("开始提取", key="btn_qty_final", use_container_width=True):
                            if "清单" in qty_mode: p = "请识别文档中所有的关键清单或表格数据，并将其整理为Markdown表格输出。"
                            elif "大纲" in qty_mode: p = "请梳理文档的逻辑结构与章节大纲，帮助我快速建立知识索引。"
                            else: p = "请提取文档中定义的核心术语、缩略语或专有名词，形成一份术语表。"
                            quick_ask(p, "analysis")
            # --- C4: 深度解读 ---
            with c4:
                with st.container(border=True):
                    st.markdown("###### 💡 **深度解读**")
                    with st.popover("🧠 配置", use_container_width=True):
                        st.markdown("###### 👤 选择解读视角")
                        role_mode = st.radio("专家角色", ["🎓 导师/教练", "⚖️ 评审/审计", "🔮 规划师/架构师"], key="role_mode_final")
                        st.divider()
                        if st.button("获取解读", key="btn_role_final", use_container_width=True):
                            if "导师" in role_mode: p = "请你扮演一位资深导师，基于这份文档，为我编写一份实操指南（How-to Guide）。"
                            elif "评审" in role_mode: p = "请你扮演一位严格的评审专家，客观评价这份文档的质量，指出其亮点与不足。"
                            else: p = "请你扮演一位顶层架构师，分析这份文档背后的设计逻辑、技术架构或政策导向。"
                            quick_ask(p, "chat")
            st.markdown("---")

        # 如果有聊天历史，则显示聊天记录 (恢复完整功能)
        else:
            chat_container = st.container(height=600)
            with chat_container:
                for msg in current_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"], unsafe_allow_html=True)
                        if "sources" in msg and msg["sources"]:
                            with st.expander(f"📚 引用了 {len(msg['sources'])} 处文档", expanded=False):
                                for idx, src in enumerate(msg["sources"]):
                                    fname = src.get('metadata', {}).get('source_file', '未知')
                                    st.markdown(f"**{idx + 1}. {fname}**")
                                    if 'image_path' in src.get('metadata', {}):
                                        img_path = src['metadata']['image_path']
                                        if os.path.exists(img_path): st.image(img_path, width=300)
                                    st.caption(src['content'][:150] + "...")
                        if "file_generated" in msg:
                            f_info = msg["file_generated"]
                            if os.path.exists(f_info["path"]):
                                with open(f_info["path"], "rb") as f:
                                    st.download_button(label=f"📥 下载：{f_info['name']}", data=f,
                                                       file_name=f_info['name'])

        # --- D. 统一的聊天输入框 ---
        if prompt := st.chat_input("有什么可以帮你的？"):
            current_messages.append({"role": "user", "content": prompt})
            st.rerun()

    # -------------------------------------------------------------
    # 4. RAG响应逻辑 (位置不变，仅需适配 chat_col)
    # -------------------------------------------------------------
    if current_messages and current_messages[-1]["role"] == "user":
        with chat_col:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response, sources, generated_file = "", [], None
                placeholder.markdown("🤔 *DeepSeek 正在思考...*")
                try:
                    if "rag_service" in st.session_state:
                        # 核心：从 session_state 获取统一的复杂配置
                        config = st.session_state.get("search_config")
                        # ⚠️ 后端 rag_service.chat_stream 需要能解析这个复杂 config
                        gen = st.session_state.rag_service.chat_stream(
                            query=current_messages[-1]["content"],
                            history=current_messages[:-1],
                            filter_config=config  # 传递整个配置字典
                        )
                        for evt in gen:
                            if evt['type'] == 'text':
                                full_response += evt['data']; placeholder.markdown(full_response + "▌")
                            elif evt['type'] == 'sources':
                                sources = evt['data']
                            elif evt['type'] == 'file':
                                generated_file = evt['data']

                        placeholder.markdown(full_response)
                        final_msg = {"role": "assistant", "content": full_response, "sources": sources}
                        if generated_file: final_msg["file_generated"] = generated_file
                        current_messages.append(final_msg)
                        st.rerun()
                except Exception as e:
                    placeholder.error(f"生成出错: {e}")

# === Tab 2: 星系图谱 (新增) ===
with tab_graph:
    # 检查是否有数据
    if "graph_data" not in st.session_state or not st.session_state.graph_data.get("nodes"):
        st.info("👋 请先在侧边栏底部的“🕸️ 知识图谱生成”区域配置并生成图谱")
        st.markdown("---")
        st.image("https://img.icons8.com/clouds/200/network.png", width=200)
    else:
        g_data = st.session_state.graph_data
        nodes_data = g_data["nodes"]
        edges_data = g_data["edges"]

        # --- 1. 三栏布局 (左控 : 中图 : 右显) ---
        col_ctrl, col_graph, col_detail = st.columns([1, 3, 1.2])

        # --- 2. 左侧：控制与搜索面板 ---
        with col_ctrl:
            st.markdown("### 🎛️ 控制台")

            # A. 统计看板
            with st.container(border=True):
                st.caption("📊 图谱统计")
                c1, c2 = st.columns(2)
                c1.metric("实体", len(nodes_data))
                c2.metric("关系", len(edges_data))

            # B. 布局设置
            st.write("")
            st.caption("📐 布局算法")
            layout_type = st.radio("视图模式", ["🕸️ 力导向 (自由)", "🌳 层级树 (分层)"], index=0, key="layout_radio")

            # C. 搜索功能
            st.write("")
            st.caption("🔍 节点搜索")
            search_term = st.text_input("查找实体", placeholder="输入关键词...", key="graph_search").strip()

            # 搜索反馈
            if search_term:
                match_count = sum(1 for n in nodes_data if search_term in n["id"])
                if match_count > 0:
                    st.success(f"匹配 {match_count} 个节点")
                else:
                    st.warning("未找到匹配节点")

        # --- 3. 中间：图谱渲染区域 ---
        with col_graph:
            vis_nodes = []
            vis_edges = []

            # 处理节点 (支持搜索高亮)
            for n in nodes_data:
                # 兼容旧版数据的兜底逻辑
                nid = n.get("id", "未知")
                label = n.get("label", nid)  # 优先用 label，没有则用 id
                color = n.get("color", "#999999")
                size = n.get("size", 25)
                desc = n.get("desc") or n.get("title") or ""

                # 💡 搜索高亮逻辑
                if search_term and search_term in nid:
                    color = "#ff0000"  # 红色高亮
                    size = 50  # 放大
                    label = f"🔍 {label}"  # 加标记

                vis_nodes.append(Node(
                    id=nid,
                    label=label,
                    size=size,
                    color=color,
                    title=desc  # 鼠标悬停显示
                ))

            # 处理边
            for e in edges_data:
                vis_edges.append(Edge(
                    source=e["source"],
                    target=e["target"],
                    label=e.get("label", ""),
                    color="#cccccc",
                    font={"align": "middle", "size": 10}
                ))

            # 配置 Config
            is_hierarchical = (layout_type == "🌳 层级树 (分层)")

            config = Config(
                width="100%",
                height=700,
                directed=True,
                physics=not is_hierarchical,  # 层级模式下关闭物理引擎
                hierarchical=is_hierarchical,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False
            )

            st.caption(f"🌌 当前展示：{st.session_state.get('graph_source_name', '未知来源')}")
            # 核心渲染组件：捕获返回值 (即被点击的节点ID)
            selected_node_id = agraph(nodes=vis_nodes, edges=vis_edges, config=config)

        # --- 4. 右侧：详情面板 ---
        with col_detail:
            st.markdown("### 📝 属性详情")

            if selected_node_id:
                # 查找选中节点的数据
                target_node = next((n for n in nodes_data if n["id"] == selected_node_id), None)

                if target_node:
                    with st.container(border=True):
                        # 标题头
                        st.markdown(f"#### {target_node.get('label', target_node['id'])}")
                        st.caption(f"ID: {target_node['id']}")
                        st.divider()

                        # 描述信息
                        st.markdown("**📄 描述**")
                        desc_text = target_node.get("desc") or target_node.get("title") or "暂无描述"
                        st.info(desc_text)

                        # 关联统计 (实时计算)
                        st.markdown("**🔗 关联统计**")
                        related_edges = [e for e in edges_data if
                                         e["source"] == selected_node_id or e["target"] == selected_node_id]
                        st.write(f"连接数: {len(related_edges)}")

                        # 列出相邻节点
                        st.markdown("**🤝 相邻实体**")
                        for i, e in enumerate(related_edges):
                            if i >= 5:  # 最多显示5个
                                st.caption("...")
                                break
                            if e["source"] == selected_node_id:
                                st.text(f"➡️ {e.get('label', '')} -> {e['target']}")
                            else:
                                st.text(f"⬅️ {e.get('label', '')} <- {e['source']}")
                else:
                    st.error("数据同步错误")
            else:
                # 未选中时的空状态
                st.info("👆 请在左侧图谱中点击任意节点，查看详细属性。")
