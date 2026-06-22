"""
Streamlit 卡片 UI —— 水厂报修智能分类系统

单文件应用，用 st.container(border=True) + 自定义 CSS 实现卡片布局。
"""
import sys
import json
from pathlib import Path

# 将项目根目录加入 sys.path，确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from data.seed_data import PLANTS, get_plant_by_name
from agent.orchestrator import RepairAgent
from agent.tools import get_all_work_orders, _work_orders, create_tools
from config import get_vector_store, get_llm

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="水厂报修智能分类系统",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义 CSS 卡片样式
# ============================================================
st.markdown("""
<style>
    /* 整体字体 */
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    }

    /* 卡片容器增强 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }

    /* 紧急程度徽章 */
    .badge-urgent {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        color: white;
    }
    .badge-紧急 { background: #EF4444; }
    .badge-一般  { background: #F59E0B; color: #1a1a1a; }
    .badge-低    { background: #10B981; }

    /* 分类标签 */
    .category-tag {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        background: #E0F2FE;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }

    /* 置信度 */
    .confidence-high { color: #059669; font-weight: 700; }
    .confidence-mid  { color: #D97706; font-weight: 700; }
    .confidence-low  { color: #DC2626; font-weight: 700; }

    /* 页头 */
    .app-header {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .app-header h1 { color: white !important; margin: 0; font-size: 28px; }
    .app-header p  { color: #BFDBFE; margin: 4px 0 0; font-size: 14px; }

    /* 相似案例条目 */
    .case-item {
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #3B82F6;
        background: #F8FAFC;
        border-radius: 0 8px 8px 0;
    }
    .case-item .score {
        font-size: 13px;
        color: #6B7280;
    }

    /* 工单状态 */
    .status-待处理 { color: #F59E0B; font-weight: 600; }
    .status-处理中 { color: #3B82F6; font-weight: 600; }
    .status-已完成 { color: #10B981; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State 初始化
# ============================================================
DEFAULTS = {
    "agent_ready": False,
    "last_result": None,
    "last_description": "",
    "last_plant": "",
    "processing": False,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================
# 延迟初始化 Agent（首次使用时才加载模型）
# ============================================================
@st.cache_resource
def get_agent():
    """缓存 Agent 实例，只初始化一次"""
    vs = get_vector_store()
    # 首次运行：写入向量数据
    if vs.is_empty():
        from data.seed_data import build_search_texts
        items = build_search_texts()
        vs.add_texts(
            [it["text"] for it in items],
            [{"id": it["id"], **it["metadata"]} for it in items],
        )
    return RepairAgent()


def friendly_error(error: Exception) -> str:
    """将技术异常转为中文友好提示"""
    msg = str(error).lower()

    if "api_key" in msg or "auth" in msg or "unauthorized" in msg or "401" in msg:
        return "🔑 API Key 无效或过期，请检查 Secrets 中的 DEEPSEEK_API_KEY。"
    if "insufficient" in msg or "balance" in msg or "quota" in msg or "402" in msg or "429" in msg:
        return "💰 API 额度不足，请前往 platform.deepseek.com 充值。"
    if "timeout" in msg or "timed out" in msg or "connect" in msg:
        return "🌐 API 连接超时，请检查网络或稍后重试。"
    if "rate" in msg or "limit" in msg:
        return "⏳ 请求过于频繁，请稍等几秒后再试。"
    if "json" in msg and "serializable" in msg:
        return "🔧 数据序列化错误，请刷新页面重试。"
    return f"❌ {str(error)[:300]}"


# ============================================================
# 辅助组件
# ============================================================
def render_confidence(value: float) -> str:
    """根据置信度返回带颜色的 HTML"""
    pct = f"{value * 100:.0f}%"
    if value >= 0.8:
        return f'<span class="confidence-high">{pct}</span>'
    elif value >= 0.5:
        return f'<span class="confidence-mid">{pct}</span>'
    else:
        return f'<span class="confidence-low">{pct}</span>'


def render_urgency_badge(urgency: str) -> str:
    """渲染紧急程度徽章"""
    return f'<span class="badge-urgent badge-{urgency}">{urgency}</span>'


def render_status_badge(status: str) -> str:
    """渲染工单状态"""
    emoji = {"待处理": "🟡", "处理中": "🔵", "已完成": "🟢"}
    return f'{emoji.get(status, "⚪")} <span class="status-{status}">{status}</span>'


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=64)
    st.title("💧 系统设置")

    st.divider()
    st.subheader("📋 示例报修")
    examples = {
        "水泵故障": "城北水厂3号送水泵轴承异响严重，振动值超过标准3倍达到12mm/s，温度升高至85°C，该泵为日供水8万吨主力机组。",
        "管道漏水": "东郊水厂DN600出厂水主管爆裂，大量自来水涌出淹没厂区道路，出厂水压力骤降50%，影响下游3个加压站。",
        "水质异常": "高新区水厂清水池出水浊度突然升高至2.8NTU，在线监测仪连续报警。原水取自水库，近日降雨导致原水浊度大幅上升。",
        "电气故障": "西区水厂低压配电室2号馈线柜断路器频繁跳闸，复位后运行10-30分钟再次跳闸，沉淀池排泥系统停运。",
        "阀门故障": "城南水厂滤池进水电动蝶阀无法远程操作，DCS显示阀门状态为故障，该阀门每天需开关4次进行反冲洗。",
    }
    for label, text in examples.items():
        if st.button(f"📌 {label}", key=f"ex_{label}", use_container_width=True):
            st.session_state["example"] = text
            st.session_state["example_plant"] = text[:4] if text[:4] in [p["name"][:2] for p in PLANTS] else ""

    st.divider()
    st.subheader("📊 工单概览")
    orders = get_all_work_orders()
    if orders:
        for o in orders[-5:]:
            st.caption(f"{o['id']} | {o['category']} | {render_status_badge(o['status'])}", unsafe_allow_html=True)
        st.metric("总工单数", len(orders))
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()
    else:
        st.caption("暂无工单，提交报修后将在此显示")

    st.divider()
    # 嵌入模式指示
    agent = get_agent()
    mode = agent.embedding_mode
    if "transformer" in mode.lower():
        st.success("🧠 嵌入模式: 语义模型", icon="🧠")
    else:
        st.warning("⚡ 嵌入模式: 轻量匹配", icon="⚡")

    st.divider()
    st.caption("🏭 水厂报修智能分类系统 v1.0")
    st.caption("Powered by LangChain + FAISS + DeepSeek")


# ============================================================
# 页面标题
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>💧 水厂报修智能分类系统</h1>
    <p>AI-Powered Repair Report Triage —— 基于 RAG 知识库的报修自动分类与工单管理</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 主布局：两列卡片
# ============================================================
left_col, right_col = st.columns(2, gap="medium")

# ---- 左侧：输入卡片 ----
with left_col:
    with st.container(border=True):
        st.subheader("📝 报修信息输入")

        # 水厂下拉选择
        plant_options = [""] + [p["name"] for p in PLANTS]
        default_plant = st.session_state.get("example_plant", "")
        plant_idx = 0
        if default_plant:
            for i, name in enumerate(plant_options):
                if default_plant[:2] in name:
                    plant_idx = i
                    break

        plant_name = st.selectbox(
            "选择水厂",
            plant_options,
            index=plant_idx,
            placeholder="请选择报修水厂...",
        )

        # 报修描述输入
        default_text = st.session_state.get("example", "")
        description = st.text_area(
            "报修描述",
            value=default_text,
            height=180,
            placeholder="请详细描述故障现象，例如：\n\n城北水厂3号送水泵轴承异响严重，振动值超标3倍达12mm/s，温度升高至85°C...",
            help="越详细越好，Agent 会据此进行分类和案例匹配",
        )

        # 提交按钮
        submitted = st.button(
            "🔍 提交分析",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("processing", False) or not description.strip(),
        )

        if submitted and description.strip():
            st.session_state["processing"] = True
            st.session_state["last_description"] = description
            st.session_state["last_plant"] = plant_name
            st.rerun()

    # ---- 左侧：相似案例卡片（结果出现后显示） ----
    if st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        similar_cases = result.get("similar_cases", [])

        with st.container(border=True):
            st.subheader("📚 相似历史案例 & 知识库")
            if similar_cases:
                for i, case in enumerate(similar_cases[:5]):
                    score_pct = case.get("score", 0) * 100
                    score_color = "🟢" if score_pct >= 70 else "🟡" if score_pct >= 50 else "🔴"

                    with st.expander(
                        f"{score_color} #{i+1} [{score_pct:.0f}%] {case.get('category', '未知')} — {case.get('description', '')[:50]}...",
                        expanded=(i == 0),
                    ):
                        st.caption(f"**相似度**: {score_pct:.1f}%")
                        st.text(case.get("description", "")[:300])

                        resolution = case.get("resolution", "")
                        if resolution:
                            st.markdown(f"**💡 处理方案**: {resolution[:300]}")
                        title = case.get("title", "")
                        if title:
                            st.markdown(f"**📄 来源**: {title}")
            else:
                st.info("未找到相似案例")

# ---- 右侧：分类结果 + 工单状态 ----
with right_col:
    # 处理中状态 —— 流式展示 Agent 执行步骤
    if st.session_state.get("processing"):
        agent = get_agent()
        status_container = st.empty()

        try:
            with status_container.container(border=True):
                st.markdown("### 🤖 Agent 分析中...")
                progress_placeholder = st.empty()
                steps_log = st.empty()

                step_lines = []
                final_result = None

                for event in agent.run_stream(
                    description=st.session_state["last_description"],
                    plant_name=st.session_state.get("last_plant", ""),
                ):
                    etype = event.get("type", "")
                    if etype == "tool_start":
                        step_lines.append(f"⚙️ {event['label']}")
                    elif etype == "tool_end":
                        step_lines[-1] = f"✅ {event['label']}"
                    elif etype == "thinking":
                        step_lines.append(f"💭 {event['label']}")
                    elif etype == "done":
                        final_result = event.get("result", {})

                    # 实时刷新步骤列表
                    progress_placeholder.markdown(
                        "\n".join(step_lines) if step_lines else "*正在启动...*"
                    )

                if final_result:
                    st.session_state["last_result"] = final_result

            st.session_state["processing"] = False
            st.session_state.pop("example", None)
            st.session_state.pop("example_plant", None)
            st.rerun()

        except Exception as e:
            st.error(friendly_error(e))
            st.session_state["processing"] = False

    # 分类结果卡片
    if st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        classification = result.get("classification", {})

        with st.container(border=True):
            st.subheader("🏷️ 分类结果")

            if classification:
                cat = classification.get("predicted_category", "未知")
                urgency = classification.get("urgency", "一般")
                confidence = classification.get("confidence", 0)
                reasoning = classification.get("reasoning", "")

                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <span class="category-tag">{cat}</span>
                    &nbsp;&nbsp;
                    {render_urgency_badge(urgency)}
                </div>
                <p><strong>置信度</strong>: {render_confidence(confidence)}</p>
                <p><strong>推理</strong>: {reasoning}</p>
                """, unsafe_allow_html=True)
            else:
                st.info("分类结果将在分析后显示")

        # 工单状态卡片
        with st.container(border=True):
            st.subheader("📋 工单处理状态")

            work_order = result.get("work_order", {})
            if work_order and work_order.get("id"):
                order_id = work_order["id"]
                # 获取最新状态（可能已被手动更新）
                latest = _work_orders.get(order_id, work_order)
                status = latest.get("status", work_order.get("status", "待处理"))

                st.markdown(f"""
                <table style="width:100%; font-size:14px;">
                    <tr><td><strong>工单号</strong></td><td><code>{order_id}</code></td></tr>
                    <tr><td><strong>状态</strong></td><td>{render_status_badge(status)}</td></tr>
                    <tr><td><strong>类别</strong></td><td>{work_order.get('category', '-')}</td></tr>
                    <tr><td><strong>指派人</strong></td><td>👤 {work_order.get('assigned_to', '-')}</td></tr>
                    <tr><td><strong>创建时间</strong></td><td>{work_order.get('created_at', '-')}</td></tr>
                </table>
                """, unsafe_allow_html=True)

                # 状态更新按钮
                st.divider()
                st.caption("更新工单状态")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("🔵 标记处理中", key="btn_processing", use_container_width=True, disabled=(status == "处理中" or status == "已完成")):
                        agent = get_agent()
                        from agent.tools import create_tools as _ct
                        tools = _ct(agent._vector_store, agent._llm)
                        update_tool = [t for t in tools if t.name == "update_work_order_status"][0]
                        update_tool.invoke({"work_order_id": order_id, "new_status": "处理中"})
                        st.rerun()
                with btn_col2:
                    if st.button("🟢 标记已完成", key="btn_done", use_container_width=True, disabled=(status == "已完成")):
                        agent = get_agent()
                        from agent.tools import create_tools as _ct
                        tools = _ct(agent._vector_store, agent._llm)
                        update_tool = [t for t in tools if t.name == "update_work_order_status"][0]
                        update_tool.invoke({"work_order_id": order_id, "new_status": "已完成"})
                        st.rerun()

        # Agent 完整回复
        with st.container(border=True):
            st.subheader("💬 Agent 分析报告")
            output = result.get("output", "")
            if output:
                st.markdown(output)
            else:
                st.info("等待分析完成...")

    elif not st.session_state.get("processing"):
        # 初始状态占位
        with st.container(border=True):
            st.subheader("🏷️ 分类结果")
            st.info("👈 请先在左侧输入报修信息并提交分析")

        with st.container(border=True):
            st.subheader("📋 工单处理状态")
            st.info("分析完成后此处将显示工单信息")


# ============================================================
# 底部：水厂联系方式（全宽）
# ============================================================
st.divider()

with st.container(border=True):
    st.subheader("📞 水厂联系方式")

    contact_cols = st.columns(len(PLANTS))
    for i, plant in enumerate(PLANTS):
        with contact_cols[i]:
            st.markdown(f"""
            <div style="text-align:center; padding:8px;">
                <strong>{plant['name']}</strong><br>
                <small>👤 {plant['contact_person']}</small><br>
                <small>📱 {plant['contact_phone']}</small><br>
                <small style="color:#6B7280;">📍 {plant['location']}</small>
            </div>
            """, unsafe_allow_html=True)

# 清除按钮
if st.session_state.get("last_result"):
    st.divider()
    if st.button("🔄 清空结果，开始新分析", use_container_width=True):
        for key in ["last_result", "last_description", "last_plant", "processing"]:
            st.session_state[key] = DEFAULTS[key]
        st.rerun()
