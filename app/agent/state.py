"""TypedDict state schema for agent workflow."""

from typing import TypedDict, List, Dict, Optional, Union, Literal
from datetime import datetime
import uuid

from ..tools.schemas import (
    AgentState, TimeRange, DataQualityReport, KPISummary, 
    MMMDecomposition, BrandEquityMetrics, ScenarioResult, 
    RiskClassification, Recommendation, ExplainabilityPackage, 
    HumanDecision, DiagnosticStep
)


class MMMAgentState(TypedDict):
    """State schema for MMM Agent workflow using LangGraph."""
    
    # Core session information
    session_id: str
    user_query: str
    timestamp: datetime
    
    # Analysis scope and parameters
    scope: List[str]
    time_range: Optional[TimeRange]
    
    # Analysis results (populated during workflow)
    data_quality: Optional[DataQualityReport]
    kpi_summary: Optional[KPISummary]
    mmm_decomposition: Optional[MMMDecomposition]
    brand_equity: Optional[BrandEquityMetrics]
    scenarios: List[ScenarioResult]
    risk_classification: Optional[RiskClassification]
    recommendations: List[Recommendation]
    
    # Explainability and audit
    explainability: Optional[ExplainabilityPackage]
    diagnostic_steps: List[DiagnosticStep]
    
    # Human-in-the-Loop
    requires_approval: bool
    hitl_status: Literal["pending", "approved", "rejected", "not_required"]
    human_decision: Optional[HumanDecision]
    
    # Workflow control
    current_step: Literal[
        "intake", "validation", "planning", "analysis", 
        "diagnosis", "scenarios", "risk_assessment", "hitl_gate",
        "recommendations", "explainability", "completed", "error"
    ]
    confidence_score: float
    error_message: Optional[str]
    completed: bool
    
    # Tool execution tracking
    tools_executed: List[str]
    execution_log: List[Dict[str, Union[str, datetime]]]


def create_initial_state(user_query: str, scope: Optional[List[str]] = None) -> MMMAgentState:
    """
    Create initial agent state for a new session.
    
    Args:
        user_query: The user's query
        scope: Optional list of SKU IDs (extracted from query if not provided)
        
    Returns:
        Initial MMMAgentState
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    return MMMAgentState(
        session_id=session_id,
        user_query=user_query,
        timestamp=timestamp,
        scope=scope or [],
        time_range=None,
        data_quality=None,
        kpi_summary=None,
        mmm_decomposition=None,
        brand_equity=None,
        scenarios=[],
        risk_classification=None,
        recommendations=[],
        explainability=None,
        diagnostic_steps=[],
        requires_approval=False,
        hitl_status="not_required",
        human_decision=None,
        current_step="intake",
        confidence_score=0.0,
        error_message=None,
        completed=False,
        tools_executed=[],
        execution_log=[]
    )


def update_state_step(state: MMMAgentState, step: str, **kwargs) -> MMMAgentState:
    """
    Update state with new step and additional data.
    
    Args:
        state: Current agent state
        step: New step name
        **kwargs: Additional fields to update
        
    Returns:
        Updated agent state
    """
    state["current_step"] = step
    state["execution_log"].append({
        "step": step,
        "timestamp": datetime.now(),
        **kwargs
    })
    
    # Update any additional fields
    for key, value in kwargs.items():
        if key in state:
            state[key] = value
    
    return state


def add_diagnostic_step(state: MMMAgentState, 
                       step_number: int,
                       action: str,
                       inputs: Dict[str, Union[str, List[str]]],
                       outputs: Dict[str, Union[str, float, List[str]]]) -> MMMAgentState:
    """
    Add a diagnostic step to the explainability trail.
    
    Args:
        state: Current agent state
        step_number: Step number
        action: Action taken
        inputs: Inputs to the action
        outputs: Outputs from the action
        
    Returns:
        Updated agent state
    """
    diagnostic_step = DiagnosticStep(
        step_number=step_number,
        action=action,
        inputs=inputs,
        outputs=outputs
    )
    
    state["diagnostic_steps"].append(diagnostic_step)
    return state


def calculate_overall_confidence(state: MMMAgentState) -> float:
    """
    Calculate overall confidence score based on available data.
    
    Args:
        state: Current agent state
        
    Returns:
        Overall confidence score (0-1)
    """
    confidence_factors = []
    
    # Data quality confidence
    if state["data_quality"]:
        confidence_factors.append(state["data_quality"].completeness_score)
    
    # MMM model confidence
    if state["mmm_decomposition"]:
        confidence_factors.append(state["mmm_decomposition"].confidence_score)
    
    # Brand equity confidence
    if state["brand_equity"]:
        confidence_factors.append(state["brand_equity"].confidence_level)
    
    # Scenario confidence
    if state["scenarios"]:
        scenario_confidence = [s.impact.confidence_score for s in state["scenarios"]]
        confidence_factors.extend(scenario_confidence)
    
    # Calculate weighted average
    if confidence_factors:
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
    else:
        overall_confidence = 0.0
    
    state["confidence_score"] = overall_confidence
    return overall_confidence


def is_analysis_complete(state: MMMAgentState) -> bool:
    """
    Check if analysis is complete based on current state.
    
    Args:
        state: Current agent state
        
    Returns:
        True if analysis is complete
    """
    required_elements = [
        state["data_quality"] is not None,
        state["kpi_summary"] is not None,
        state["mmm_decomposition"] is not None,
        len(state["recommendations"]) > 0,
        state["explainability"] is not None
    ]
    
    return all(required_elements) and state["current_step"] not in ["error", "hitl_gate"]


def get_state_summary(state: MMMAgentState) -> Dict[str, Union[str, float, int]]:
    """
    Get a summary of the current state for monitoring.
    
    Args:
        state: Current agent state
        
    Returns:
        State summary dictionary
    """
    return {
        "session_id": state["session_id"],
        "current_step": state["current_step"],
        "confidence_score": state["confidence_score"],
        "tools_executed_count": len(state["tools_executed"]),
        "diagnostic_steps_count": len(state["diagnostic_steps"]),
        "scenarios_count": len(state["scenarios"]),
        "recommendations_count": len(state["recommendations"]),
        "requires_approval": state["requires_approval"],
        "hitl_status": state["hitl_status"],
        "completed": state["completed"],
        "error_message": state["error_message"]
    }
