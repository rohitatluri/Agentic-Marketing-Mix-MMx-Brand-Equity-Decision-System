"""LangGraph workflow for MMM agent system."""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import MMMAgentState, create_initial_state, get_state_summary
from .nodes import MMMAgentNodes

logger = logging.getLogger(__name__)


class MMMAgentWorkflow:
    """LangGraph workflow for MMM Agent with state management and conditional routing."""
    
    def __init__(self):
        """Initialize the MMM Agent workflow."""
        self.nodes = MMMAgentNodes()
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with all nodes and edges."""
        
        # Create the workflow graph
        workflow = StateGraph(MMMAgentState)
        
        # Add all nodes
        workflow.add_node("intake", self.nodes.intake_node)
        workflow.add_node("validation", self.nodes.validation_node)
        workflow.add_node("planning", self.nodes.planning_node)
        workflow.add_node("analysis", self.nodes.analysis_node)
        workflow.add_node("diagnosis", self.nodes.diagnosis_node)
        workflow.add_node("scenarios", self.nodes.scenarios_node)
        workflow.add_node("risk_assessment", self.nodes.risk_assessment_node)
        workflow.add_node("recommendations", self.nodes.recommendations_node)
        workflow.add_node("explainability", self.nodes.explainability_node)
        workflow.add_node("completion", self.nodes.completion_node)
        workflow.add_node("hitl_gate", self._hitl_gate_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set the entry point
        workflow.set_entry_point("intake")
        
        # Add edges for the main workflow
        workflow.add_edge("intake", "validation")
        workflow.add_edge("validation", "planning")
        workflow.add_edge("planning", "analysis")
        workflow.add_edge("analysis", "diagnosis")
        
        # Conditional routing from diagnosis
        workflow.add_conditional_edges(
            "diagnosis",
            self._route_from_diagnosis,
            {
                "scenarios": "scenarios",
                "recommendations": "recommendations",
                "explainability": "explainability",
                "error": "error_handler"
            }
        )
        
        # Conditional routing from scenarios
        workflow.add_conditional_edges(
            "scenarios",
            self._route_from_scenarios,
            {
                "risk_assessment": "risk_assessment",
                "recommendations": "recommendations",
                "error": "error_handler"
            }
        )
        
        # Conditional routing from risk assessment
        workflow.add_conditional_edges(
            "risk_assessment",
            self._route_from_risk_assessment,
            {
                "hitl_gate": "hitl_gate",
                "recommendations": "recommendations",
                "error": "error_handler"
            }
        )
        
        # HITL gate routing
        workflow.add_conditional_edges(
            "hitl_gate",
            self._route_from_hitl_gate,
            {
                "recommendations": "recommendations",
                "explainability": "explainability",
                "error": "error_handler"
            }
        )
        
        # Final edges
        workflow.add_edge("recommendations", "explainability")
        workflow.add_edge("explainability", "completion")
        workflow.add_edge("completion", END)
        workflow.add_edge("error_handler", END)
        
        return workflow
    
    def _route_from_diagnosis(self, state: MMMAgentState) -> str:
        """Route from diagnosis based on query intent and analysis results."""
        try:
            # Check for errors
            if state.get("error_message"):
                return "error"
            
            # Get query intent
            query_intent = self._classify_query_intent(state["user_query"])
            
            # Route based on intent
            if query_intent == "scenario":
                return "scenarios"
            elif query_intent == "recommendation":
                return "scenarios"  # Need scenarios before recommendations
            elif query_intent == "diagnosis":
                return "explainability"  # Can go directly to explainability
            else:
                return "recommendations"  # Default path
                
        except Exception as e:
            logger.error(f"Error in diagnosis routing: {str(e)}")
            return "error"
    
    def _route_from_scenarios(self, state: MMMAgentState) -> str:
        """Route from scenarios based on scenario results."""
        try:
            # Check for errors
            if state.get("error_message"):
                return "error"
            
            # If no scenarios generated, go to recommendations
            if not state.get("scenarios"):
                return "recommendations"
            
            # Go to risk assessment for scenarios
            return "risk_assessment"
            
        except Exception as e:
            logger.error(f"Error in scenarios routing: {str(e)}")
            return "error"
    
    def _route_from_risk_assessment(self, state: MMMAgentState) -> str:
        """Route from risk assessment based on approval requirements."""
        try:
            # Check for errors
            if state.get("error_message"):
                return "error"
            
            # Check if human approval is required
            if state.get("requires_approval", False):
                return "hitl_gate"
            else:
                return "recommendations"
                
        except Exception as e:
            logger.error(f"Error in risk assessment routing: {str(e)}")
            return "error"
    
    def _route_from_hitl_gate(self, state: MMMAgentState) -> str:
        """Route from HITL gate based on human decision."""
        try:
            # Check for errors
            if state.get("error_message"):
                return "error"
            
            # Check HITL status
            hitl_status = state.get("hitl_status", "not_required")
            
            if hitl_status == "approved":
                return "recommendations"
            elif hitl_status == "rejected":
                return "explainability"  # Still provide explainability for rejected actions
            else:
                return "recommendations"  # Default case
                
        except Exception as e:
            logger.error(f"Error in HITL gate routing: {str(e)}")
            return "error"
    
    def _hitl_gate_node(self, state: MMMAgentState) -> MMMAgentState:
        """Human-in-the-Loop gate node for high-risk actions."""
        try:
            logger.info("Entering HITL gate for high-risk action")
            
            # In a real implementation, this would:
            # 1. Pause execution
            # 2. Present risk assessment to user
            # 3. Wait for human input
            # 4. Resume based on decision
            
            # For demo purposes, we'll auto-approve with a delay simulation
            hitl_status = state.get("hitl_status", "pending")
            
            if hitl_status == "pending":
                # Simulate human approval (in production, this would wait for actual human input)
                from ..tools.schemas import HumanDecision
                
                human_decision = HumanDecision(
                    approver="demo_manager",
                    action="Auto-approved for demo",
                    decision="approved",
                    reasoning="Demo mode - auto-approving for testing"
                )
                
                state = state.copy()
                state["hitl_status"] = "approved"
                state["human_decision"] = human_decision
                
                logger.info("HITL gate: Auto-approved for demo purposes")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in HITL gate: {str(e)}")
            state = state.copy()
            state["error_message"] = f"HITL gate error: {str(e)}"
            return state
    
    def _error_handler_node(self, state: MMMAgentState) -> MMMAgentState:
        """Error handler node for graceful error handling."""
        try:
            error_message = state.get("error_message", "Unknown error")
            logger.error(f"Error handler triggered: {error_message}")
            
            # Create a minimal explainability package even for errors
            from ..tools.schemas import ExplainabilityPackage
            
            explainability = ExplainabilityPackage(
                session_id=state["session_id"],
                query=state["user_query"],
                diagnostic_steps=state.get("diagnostic_steps", []),
                confidence_breakdown={"error": 0.0},
                source_attribution={"error": "workflow_failed"},
                alternatives_considered=[],
                overall_confidence=0.0
            )
            
            state = state.copy()
            state["explainability"] = explainability
            state["completed"] = True
            state["confidence_score"] = 0.0
            
            return state
            
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            return state
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the user's query intent for routing."""
        query_lower = query.lower()
        
        if "why" in query_lower and ("drop" in query_lower or "decline" in query_lower):
            return "diagnosis"
        elif "what should" in query_lower or "recommend" in query_lower:
            return "recommendation"
        elif "what if" in query_lower or "scenario" in query_lower:
            return "scenario"
        elif "compare" in query_lower:
            return "comparison"
        else:
            return "general_analysis"
    
    def run_analysis(self, user_query: str, scope: Optional[list] = None) -> Dict[str, Any]:
        """
        Run the complete MMM analysis workflow.
        
        Args:
            user_query: The user's query
            scope: Optional list of SKU IDs
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Starting MMM analysis for query: {user_query}")
            
            # Create initial state
            initial_state = create_initial_state(user_query, scope)
            
            # Compile the workflow
            app = self.workflow.compile(checkpointer=self.memory)
            
            # Run the workflow
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            result = app.invoke(initial_state, config=config)
            
            # Create response summary
            response = self._create_response_summary(result)
            
            logger.info(f"Analysis completed successfully for session: {result['session_id']}")
            return response
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return {
                "error": str(e),
                "session_id": initial_state.get("session_id", "unknown"),
                "completed": False,
                "confidence_score": 0.0
            }
    
    def _create_response_summary(self, state: MMMAgentState) -> Dict[str, Any]:
        """Create a response summary from the final state."""
        summary = {
            "session_id": state["session_id"],
            "user_query": state["user_query"],
            "completed": state.get("completed", False),
            "current_step": state.get("current_step", "unknown"),
            "confidence_score": state.get("confidence_score", 0.0),
            "error_message": state.get("error_message"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add analysis results if available
        if state.get("kpi_summary"):
            summary["kpi_summary"] = {
                "gmv_change_pct": state["kpi_summary"].gmv_change_pct,
                "units_change_pct": state["kpi_summary"].units_change_pct,
                "data_quality_score": state["kpi_summary"].data_quality_score
            }
        
        if state.get("mmm_decomposition"):
            summary["mmm_decomposition"] = {
                "total_change_pct": state["mmm_decomposition"].total_change_pct,
                "model_fit_r2": state["mmm_decomposition"].model_fit_r2,
                "driver_contributions": [
                    {
                        "driver": contrib.driver,
                        "contribution_pct": contrib.contribution_pct
                    }
                    for contrib in state["mmm_decomposition"].driver_contributions
                ]
            }
        
        if state.get("brand_equity"):
            summary["brand_equity"] = {
                "brand_equity_index": state["brand_equity"].brand_equity_index,
                "trend_direction": state["brand_equity"].trend_direction,
                "confidence_level": state["brand_equity"].confidence_level
            }
        
        if state.get("recommendations"):
            summary["recommendations"] = [
                {
                    "rank": rec.rank,
                    "action": rec.action,
                    "short_term_impact": rec.short_term_impact,
                    "long_term_impact": rec.long_term_impact,
                    "risk_level": rec.risk_level.value,
                    "confidence_score": rec.confidence_score
                }
                for rec in state["recommendations"]
            ]
        
        if state.get("scenarios"):
            summary["scenarios"] = [
                {
                    "interventions": [
                        {
                            "parameter": interv.parameter,
                            "change": interv.change,
                            "unit": interv.unit
                        }
                        for interv in scenario.interventions
                    ],
                    "short_term_gmv_impact": scenario.impact.short_term_gmv_impact,
                    "long_term_brand_impact": scenario.impact.long_term_brand_impact,
                    "risk_level": scenario.impact.risk_level.value,
                    "confidence_score": scenario.impact.confidence_score
                }
                for scenario in state["scenarios"]
            ]
        
        if state.get("explainability"):
            summary["explainability"] = {
                "diagnostic_steps_count": len(state["explainability"].diagnostic_steps),
                "confidence_breakdown": state["explainability"].confidence_breakdown,
                "overall_confidence": state["explainability"].overall_confidence
            }
        
        # Add HITL information if applicable
        if state.get("requires_approval"):
            summary["hitl_required"] = True
            summary["hitl_status"] = state.get("hitl_status", "unknown")
            if state.get("human_decision"):
                summary["human_decision"] = {
                    "decision": state["human_decision"].decision,
                    "approver": state["human_decision"].approver,
                    "reasoning": state["human_decision"].reasoning
                }
        
        return summary
    
    def get_workflow_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow session."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state_snapshot = self.memory.get(config)
            
            if state_snapshot and "values" in state_snapshot:
                return get_state_summary(state_snapshot["values"])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting workflow state: {str(e)}")
            return None
    
    def visualize_workflow(self) -> str:
        """Generate a text representation of the workflow for debugging."""
        return """
        MMM Agent Workflow:
        
        intake → validation → planning → analysis → diagnosis
                                                          ↓
                                                    scenarios → risk_assessment
                                                          ↓                    ↓
                                                    recommendations ← hitl_gate
                                                          ↓
                                                explainability → completion → END
        """
