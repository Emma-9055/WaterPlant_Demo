"""
Pydantic 数据模型：水厂、报修分类、历史案例、分类结果、工单
"""
from pydantic import BaseModel, Field
from typing import Optional


class WaterPlant(BaseModel):
    """水厂信息"""
    id: str
    name: str
    location: str
    contact_person: str
    contact_phone: str


class RepairCategory(BaseModel):
    """报修类别"""
    id: str
    name: str
    typical_duration_hours: float
    default_urgency: str  # 紧急 / 一般 / 低


class HistoricalCase(BaseModel):
    """历史报修案例"""
    id: str
    plant_name: str
    category: str
    urgency: str
    description: str
    resolution: str
    resolution_time_hours: float
    date: str


class ClassificationResult(BaseModel):
    """Agent 分类结果"""
    predicted_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    urgency: str
    reasoning: str


class SimilarCase(BaseModel):
    """向量检索返回的相似案例"""
    case_id: str
    description: str
    resolution: str
    similarity_score: float


class WorkOrder(BaseModel):
    """工单"""
    id: str
    report_description: str
    category: str = ""
    urgency: str = ""
    plant_name: str = ""
    status: str = "待处理"  # 待处理 / 处理中 / 已完成
    assigned_to: str = ""
    created_at: str = ""
    notes: str = ""
