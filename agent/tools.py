"""
LangChain Agent 工具集（5 个工具）

工具通过 create_tools() 工厂函数创建，依赖注入向量存储和 LLM。
"""
import json
import re
import numpy as np
from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool

# 北京时间（UTC+8）
BEIJING_TZ = timezone(timedelta(hours=8))

from data.seed_data import PLANTS, CATEGORIES, get_plant_by_name

# 内存工单存储（demo 级别，重启清空）
_work_orders: dict[str, dict] = {}


def _safe_json_dumps(obj) -> str:
    """安全 JSON 序列化，自动转换 numpy 类型"""

    def _convert(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_convert(i) for i in o]
        return o

    return json.dumps(_convert(obj), ensure_ascii=False)


def _classify_with_llm(description: str, llm) -> dict:
    """调用 LLM 进行报修分类，返回结构化结果"""
    categories_desc = "\n".join(
        f"- {c['name']}（默认紧急程度: {c['default_urgency']}，通常修复时间: {c['typical_duration_hours']}h）"
        for c in CATEGORIES
    )

    prompt = f"""你是一个自来水厂维修报告分类专家。请分析以下报修描述，给出分类结果。

## 可选类别
{categories_desc}

## 紧急程度标准
- 紧急：直接影响供水安全、大范围停水、水质超标、设备爆裂
- 一般：设备异常但未停机、小范围渗漏、非关键设备故障
- 低：轻微异常、预防性维护相关

## 报修描述
{description}

## 输出要求
请严格按以下 JSON 格式输出，不要包含任何其他文字：
{{"predicted_category": "类别名称（5选1）", "confidence": 0.0-1.0, "urgency": "紧急/一般/低", "reasoning": "分类理由（50字以内）"}}"""

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    # 尝试从回复中提取 JSON
    try:
        # 提取第一个 {...} 块
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # 解析失败时的 fallback
    return {
        "predicted_category": "未分类",
        "confidence": 0.0,
        "urgency": "一般",
        "reasoning": f"LLM 返回无法解析: {content[:100]}",
    }


def create_tools(vector_store, llm):
    """工厂函数：创建所有 Agent 工具（依赖注入）"""

    @tool
    def classify_repair_report(description: str) -> str:
        """
        对维修报告进行分类，判断报修类别和紧急程度。
        输入：报修描述的完整文本。
        返回：JSON 格式的分类结果，包含 predicted_category（类别）、confidence（置信度）、urgency（紧急程度）和 reasoning（推理）。
        在收到用户报修描述后应首先调用此工具。
        """
        result = _classify_with_llm(description, llm)
        return _safe_json_dumps(result)

    @tool
    def search_similar_cases(query: str, k: int = 5) -> str:
        """
        在历史案例和维修知识库中搜索与查询内容最相似的记录。
        输入：query - 搜索查询文本（建议用报修描述的关键部分），k - 返回结果数量。
        返回：JSON 格式的相似案例列表，每项包含 id、相似度分数、描述和处理方案。
        用于获取类似故障的历史处理经验和维修指南。
        """
        results = vector_store.search(query, k=k)
        simplified = []
        for r in results:
            simplified.append({
                "id": r["id"],
                "score": r["score"],
                "description": r["text"][:200],
                "resolution": r.get("metadata", {}).get("resolution", ""),
                "title": r.get("metadata", {}).get("title", ""),
                "category": r.get("metadata", {}).get("category", ""),
            })
        return _safe_json_dumps(simplified)

    @tool
    def create_work_order(
        report_description: str,
        category: str,
        urgency: str,
        plant_name: str,
    ) -> str:
        """
        创建维修工单。
        输入：
        - report_description: 原始报修描述
        - category: 报修类别（来自 classify_repair_report 的结果）
        - urgency: 紧急程度
        - plant_name: 水厂名称
        返回：JSON 格式的已创建工单信息，包含工单 ID 和状态。
        对于紧急报修或用户明确要求时调用此工具。
        """
        order_id = f"WO-{datetime.now(BEIJING_TZ).strftime('%Y%m%d')}-{len(_work_orders) + 1:03d}"
        plant = get_plant_by_name(plant_name)
        assigned_to = plant["contact_person"] if plant else "待分配"

        order = {
            "id": order_id,
            "report_description": report_description,
            "category": category,
            "urgency": urgency,
            "plant_name": plant_name,
            "status": "待处理",
            "assigned_to": assigned_to,
            "created_at": datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "",
        }
        _work_orders[order_id] = order
        return _safe_json_dumps(order)

    @tool
    def get_plant_contact(plant_name: str) -> str:
        """
        查找水厂的联系方式。
        输入：plant_name - 水厂名称（支持模糊匹配，如"城北"可匹配"城北水厂"）。
        返回：JSON 格式的水厂信息，包含联系人、电话、地址。
        当需要提供水厂联系方式或分配工单时调用。
        """
        plant = get_plant_by_name(plant_name)
        if plant:
            return _safe_json_dumps(plant)
        # 列出所有水厂
        all_plants = [{"name": p["name"], "contact_person": p["contact_person"], "contact_phone": p["contact_phone"]} for p in PLANTS]
        return _safe_json_dumps({"error": f"未找到匹配 '{plant_name}' 的水厂", "available_plants": all_plants})

    @tool
    def update_work_order_status(work_order_id: str, new_status: str) -> str:
        """
        更新工单处理状态。
        输入：work_order_id - 工单号（如 WO-20260622-001），new_status - 新状态（待处理/处理中/已完成）。
        返回：JSON 格式的更新后工单信息。
        """
        if work_order_id not in _work_orders:
            return _safe_json_dumps({"error": f"工单 {work_order_id} 不存在"})
        valid_statuses = ["待处理", "处理中", "已完成"]
        if new_status not in valid_statuses:
            return _safe_json_dumps({"error": f"无效状态 '{new_status}'，可选: {valid_statuses}"})
        _work_orders[work_order_id]["status"] = new_status
        return _safe_json_dumps(_work_orders[work_order_id])

    return [
        classify_repair_report,
        search_similar_cases,
        create_work_order,
        get_plant_contact,
        update_work_order_status,
    ]


def get_all_work_orders() -> list[dict]:
    """获取所有工单（供 UI 查询）"""
    return list(_work_orders.values())


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    from config import get_llm, get_vector_store

    vs = get_vector_store()
    llm = get_llm()
    tools = create_tools(vs, llm)

    print("可用工具:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:80]}...")

    # 确保向量库有数据
    if vs.is_empty():
        from data.seed_data import build_search_texts
        items = build_search_texts()
        vs.add_texts(
            [it["text"] for it in items],
            [{"id": it["id"], **it["metadata"]} for it in items],
        )

    # 测试分类
    print("\n--- 测试 classify_repair_report ---")
    result = tools[0].invoke("城北水厂3号水泵轴承异响严重，振动值超标")
    print(result)

    # 测试搜索
    print("\n--- 测试 search_similar_cases ---")
    result = tools[1].invoke("水泵异响振动大")
    print(result[:300])

    # 测试联系方式
    print("\n--- 测试 get_plant_contact ---")
    result = tools[3].invoke("城北")
    print(result)
