"""Pydantic models for data validation and type safety."""

from datetime import datetime, date
from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class RiskLevel(str, Enum):
    """Risk classification for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendDirection(str, Enum):
    """Trend direction for metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class TimeRange(BaseModel):
    """Time range specification."""
    start_date: date
    end_date: date
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class KPISummary(BaseModel):
    """KPI summary for SKU(s) over time range."""
    scope: List[str] = Field(..., description="List of SKU IDs")
    time_range: TimeRange
    gmv_current: float = Field(..., ge=0, description="Current period GMV")
    gmv_previous: float = Field(..., ge=0, description="Previous period GMV")
    gmv_change_pct: float = Field(..., description="GMV change percentage")
    units_current: int = Field(..., ge=0, description="Current period units")
    units_previous: int = Field(..., ge=0, description="Previous period units")
    units_change_pct: float = Field(..., description="Units change percentage")
    aov_current: float = Field(..., ge=0, description="Current period AOV")
    aov_previous: float = Field(..., ge=0, description="Previous period AOV")
    aov_change_pct: float = Field(..., description="AOV change percentage")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    anomaly_flags: List[str] = Field(default_factory=list, description="Anomaly flags")


class DriverContribution(BaseModel):
    """Individual driver contribution to sales change."""
    driver: str = Field(..., description="Driver name (e.g., 'SLA', 'Price', 'Marketing')")
    contribution_pct: float = Field(..., description="Contribution percentage")
    impact_amount: float = Field(..., description="Impact amount in currency")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")


class MMMDecomposition(BaseModel):
    """Marketing Mix Model decomposition results."""
    scope: List[str]
    time_range: TimeRange
    total_change_pct: float = Field(..., description="Total change percentage")
    baseline_demand: float = Field(..., description="Baseline demand level")
    residual: float = Field(..., description="Unexplained variation")
    model_fit_r2: float = Field(..., ge=0, le=1, description="Model R-squared")
    driver_contributions: List[DriverContribution]
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence score")


class BrandEquityMetrics(BaseModel):
    """Brand equity index and related metrics."""
    scope: List[str]
    time_range: TimeRange
    brand_equity_index: float = Field(..., ge=0, le=100, description="Brand Equity Index (0-100)")
    previous_index: float = Field(..., ge=0, le=100, description="Previous period index")
    trend_direction: TrendDirection = Field(..., description="Trend direction")
    velocity: float = Field(..., description="Change per week")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence level")
    risk_alerts: List[str] = Field(default_factory=list, description="Risk alerts")


class Intervention(BaseModel):
    """Proposed intervention for scenario simulation."""
    parameter: str = Field(..., description="Parameter to change (e.g., 'discount', 'sla')")
    change: float = Field(..., description="Change amount (can be negative)")
    unit: str = Field(..., description="Unit of change (e.g., '%', 'days')")


class ScenarioImpact(BaseModel):
    """Impact of a scenario simulation."""
    short_term_gmv_impact: float = Field(..., description="Short-term GMV impact (4 weeks)")
    long_term_brand_impact: float = Field(..., description="Long-term brand impact (12 weeks)")
    net_roi: float = Field(..., description="Net ROI over 12 weeks")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")
    uncertainty_band: Optional[Dict[str, float]] = Field(None, description="Uncertainty bounds")


class ScenarioResult(BaseModel):
    """Complete scenario simulation result."""
    interventions: List[Intervention]
    impact: ScenarioImpact
    reasoning: str = Field(..., description="Explanation of the scenario logic")


class DataQualityIssue(BaseModel):
    """Individual data quality issue."""
    issue_type: str = Field(..., description="Type of issue")
    description: str = Field(..., description="Description of the issue")
    severity: Literal["low", "medium", "high"] = Field(..., description="Severity level")
    affected_weeks: List[str] = Field(default_factory=list, description="Affected weeks")


class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    scope: List[str]
    time_range: TimeRange
    completeness_score: float = Field(..., ge=0, le=1, description="Data completeness score")
    missing_weeks: List[str] = Field(default_factory=list, description="Missing weeks")
    anomalies: List[DataQualityIssue] = Field(default_factory=list, description="Detected anomalies")
    supply_suppression_flags: List[str] = Field(default_factory=list, description="Supply suppression flags")
    recommendation: Literal["proceed", "caution", "insufficient_data"] = Field(..., description="Recommendation")


class RiskClassification(BaseModel):
    """Risk classification for an action."""
    action: str = Field(..., description="Proposed action description")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    reasoning: str = Field(..., description="Reasoning for classification")
    approval_required: bool = Field(..., description="Whether human approval is required")
    hitl_message: Optional[str] = Field(None, description="Suggested HITL message to user")
    short_term_gain: Optional[float] = Field(None, description="Expected short-term gain")
    long_term_cost: Optional[float] = Field(None, description="Expected long-term cost")


class Recommendation(BaseModel):
    """Action recommendation."""
    rank: int = Field(..., ge=1, description="Recommendation rank")
    action: str = Field(..., description="Recommended action")
    short_term_impact: str = Field(..., description="Short-term impact description")
    long_term_impact: str = Field(..., description="Long-term impact description")
    risk_level: RiskLevel = Field(..., description="Risk level")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")
    feasibility: str = Field(..., description="Feasibility assessment")
    caveats: List[str] = Field(default_factory=list, description="Important caveats")


class DiagnosticStep(BaseModel):
    """Step in diagnostic reasoning chain."""
    step_number: int = Field(..., ge=1, description="Step number")
    action: str = Field(..., description="Action taken")
    inputs: Dict[str, Union[str, List[str]]] = Field(..., description="Inputs to the action")
    outputs: Dict[str, Union[str, float, List[str]]] = Field(..., description="Outputs from the action")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of step")


class ExplainabilityPackage(BaseModel):
    """Complete explainability package for audit trail."""
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="Original user query")
    diagnostic_steps: List[DiagnosticStep] = Field(..., description="Step-by-step reasoning")
    confidence_breakdown: Dict[str, float] = Field(..., description="Confidence breakdown")
    source_attribution: Dict[str, str] = Field(..., description="Source attribution for claims")
    alternatives_considered: List[str] = Field(..., description="Alternative approaches considered")
    overall_confidence: float = Field(..., ge=0, le=1, description="Overall confidence score")


class HumanDecision(BaseModel):
    """Human decision record for HITL."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Decision timestamp")
    approver: str = Field(..., description="Approver name/ID")
    action: str = Field(..., description="Action being approved/rejected")
    decision: Literal["approved", "rejected", "modified"] = Field(..., description="Human decision")
    reasoning: str = Field(..., description="Human reasoning for decision")
    modifications: Optional[Dict[str, Union[str, float]]] = Field(None, description="Modifications if any")


class AgentState(BaseModel):
    """Complete agent state for LangGraph workflow."""
    session_id: str = Field(..., description="Session identifier")
    user_query: str = Field(..., description="Original user query")
    scope: List[str] = Field(default_factory=list, description="SKU scope")
    time_range: Optional[TimeRange] = Field(None, description="Analysis time range")
    data_quality: Optional[DataQualityReport] = Field(None, description="Data quality assessment")
    kpi_summary: Optional[KPISummary] = Field(None, description="KPI summary")
    mmm_decomposition: Optional[MMMDecomposition] = Field(None, description="MMM decomposition")
    brand_equity: Optional[BrandEquityMetrics] = Field(None, description="Brand equity metrics")
    scenarios: List[ScenarioResult] = Field(default_factory=list, description="Scenario results")
    risk_classification: Optional[RiskClassification] = Field(None, description="Risk classification")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Recommendations")
    explainability: Optional[ExplainabilityPackage] = Field(None, description="Explainability package")
    human_decision: Optional[HumanDecision] = Field(None, description="Human decision if HITL triggered")
    requires_approval: bool = Field(default=False, description="Whether human approval is required")
    hitl_status: Literal["pending", "approved", "rejected", "not_required"] = Field("not_required", description="HITL status")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Overall confidence")
    error_message: Optional[str] = Field(None, description="Error message if any")
    completed: bool = Field(default=False, description="Whether analysis is completed")
    
    class Config:
        arbitrary_types_allowed = True
