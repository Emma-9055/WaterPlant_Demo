"""
LangChain Agent 编排层

封装 ReAct Agent 的创建和执行，提供干净的 run() 接口。
使用 langgraph.prebuilt.create_react_agent 以兼容 LangChain v1.x。
"""
import json
import re
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from config import get_llm, get_vector_store
from agent.tools import create_tools

# ============================================================
# 中文 System Prompt
# ============================================================
SYSTEM_PROMPT = """你是一个自来水厂维修报告智能分类助手。你的职责是帮助处理报修信息。

## 你的能力
1. **分类报修**：根据报修描述判断故障类别（管道漏水、水泵故障、电气故障、水质异常、阀门故障）
2. **评估紧急程度**：判断是否需要立即处理（紧急/一般/低）
3. **检索相似案例**：在历史案例库中搜索类似故障的处理经验
4. **创建工单**：为报修创建正式工单，指派负责人
5. **查询联系方式**：提供各水厂的联系人信息

## 工作流程
当收到用户的报修描述后，你应该：
1. 首先调用 classify_repair_report 工具进行分类
2. 同时或之后调用 search_similar_cases 工具检索历史相似案例
3. 根据紧急程度判断是否需要创建工单（紧急的必须创建）
4. 如果用户提到了水厂名称，调用 get_plant_contact 提供联系方式
5. 最后总结所有信息，清晰地向用户展示结果

## 回复要求
- 使用中文回复
- 结构清晰，分点列出分类结果、相似案例、工单信息和联系方式
- 对于紧急报修，用醒目的方式提醒用户
- 如果用户没有提供足够信息（如水厂名称），主动询问
- 保持专业、简洁、有帮助的语气
"""


class RepairAgent:
    """水厂报修分类 Agent —— 封装 LangChain Agent 的创建和执行"""

    def __init__(self, llm=None, vector_store=None):
        self._llm = llm or get_llm()
        self._vector_store = vector_store or get_vector_store()
        self._tools = create_tools(self._vector_store, self._llm)
        self._agent = self._build_agent()
        self.last_result: dict = {}  # 可供 UI 读取的结构化结果

    def _build_agent(self):
        """构建 LangChain Agent（使用 langgraph）"""
        return create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self, description: str, plant_name: str = "") -> dict:
        """
        执行报修分析。

        参数:
            description: 报修描述文本
            plant_name: 可选，水厂名称

        返回:
            {
                "output": str,           # Agent 最终回复（Markdown）
                "classification": dict,  # 分类结果
                "similar_cases": list,   # 相似案例
                "work_order": dict,      # 工单信息
                "plant_contact": dict,   # 水厂联系方式
                "intermediate_steps": list,  # 中间步骤（供调试）
            }
        """
        # 构建输入
        user_input = f"报修描述：{description}"
        if plant_name:
            user_input += f"\n水厂名称：{plant_name}"

        # 执行 Agent（langgraph 用 messages 格式）
        result = self._agent.invoke({"messages": [HumanMessage(content=user_input)]})

        # 提取最终输出
        messages = result.get("messages", [])
        final_output = ""
        intermediate_steps = []

        for msg in messages:
            # 收集工具调用和结果
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    intermediate_steps.append({
                        "tool": tc.get("name", "unknown"),
                        "input": tc.get("args", {}),
                        "output": None,  # 下一个 ToolMessage 中
                    })
            if hasattr(msg, "content") and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                # ToolMessage - 关联到最近的 tool_call
                if intermediate_steps:
                    intermediate_steps[-1]["output"] = msg.content
            elif hasattr(msg, "content") and not hasattr(msg, "tool_calls"):
                # 普通 AI 消息——最终回复
                final_output = msg.content if isinstance(msg.content, str) else str(msg.content)

        if not final_output:
            # fallback: 使用最后一条消息
            last_msg = messages[-1] if messages else None
            if last_msg and hasattr(last_msg, "content"):
                final_output = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

        # 从 intermediate_steps 提取结构化数据
        parsed = self._parse_steps(intermediate_steps)
        parsed["output"] = final_output
        self.last_result = parsed
        return parsed

    def _parse_steps(self, steps: list[dict]) -> dict:
        """从工具调用步骤中提取结构化结果"""
        parsed = {
            "classification": {},
            "similar_cases": [],
            "work_order": {},
            "plant_contact": {},
        }

        for step in steps:
            tool_name = step.get("tool", "")
            output = step.get("output", "")

            if not output:
                continue

            try:
                obs_data = json.loads(output) if isinstance(output, str) else output
            except (json.JSONDecodeError, TypeError):
                obs_data = output

            if tool_name == "classify_repair_report":
                if isinstance(obs_data, dict):
                    parsed["classification"] = obs_data

            elif tool_name == "search_similar_cases":
                if isinstance(obs_data, list):
                    parsed["similar_cases"] = obs_data

            elif tool_name == "create_work_order":
                if isinstance(obs_data, dict) and "id" in obs_data:
                    parsed["work_order"] = obs_data

            elif tool_name == "get_plant_contact":
                if isinstance(obs_data, dict) and "contact_person" in obs_data:
                    parsed["plant_contact"] = obs_data

        return parsed


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    # 确保向量库已初始化
    vs = get_vector_store()
    if vs.is_empty():
        print("正在初始化向量库...")
        from data.seed_data import build_search_texts
        items = build_search_texts()
        vs.add_texts(
            [it["text"] for it in items],
            [{"id": it["id"], **it["metadata"]} for it in items],
        )
        print(f"已写入 {len(items)} 条记录")

    agent = RepairAgent()
    print("\n" + "=" * 60)
    print("测试：城北水厂3号水泵轴承异响，振动值超标3倍")
    print("=" * 60)
    result = agent.run(
        description="城北水厂3号送水泵轴承异响严重，振动值超过标准3倍达到12mm/s，温度升高至85°C。该泵为日供水8万吨的主力机组。",
        plant_name="城北水厂",
    )

    print("\n--- Agent 回复 ---")
    print(result["output"])
    print("\n--- 结构化结果 ---")
    print(f"分类: {result['classification']}")
    print(f"相似案例数: {len(result['similar_cases'])}")
    print(f"工单: {result['work_order'].get('id', '无')}")
    print(f"联系方式: {result['plant_contact'].get('contact_person', '无')}")
