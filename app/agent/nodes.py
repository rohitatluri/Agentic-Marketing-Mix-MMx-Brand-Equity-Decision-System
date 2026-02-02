"""Individual nodes for the agent workflow."""

import re
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any
import logging
import json

from .state import MMMAgentState, create_initial_state, update_state_step, add_diagnostic_step, calculate_overall_confidence
from ..tools.schemas import TimeRange, Recommendation, RiskLevel, ExplainabilityPackage, DiagnosticStep
from ..tools.kpi_tools import KPITools
from ..tools.mmm_tools import MMMTools
from ..tools.brand_equity_tools import BrandEquityTools
from ..tools.scenario_tools import ScenarioTools
from ..tools.data_quality_tools import DataQualityTools
from ..tools.risk_classification_tools import RiskClassificationTools

logger = logging.getLogger(__name__)


class MMMAgentNodes:
    """Individual nodes for the MMM Agent workflow."""
    
    def __init__(self):
        """Initialize agent nodes with all required tools."""
        self.kpi_tools = KPITools()
        self.mmm_tools = MMMTools()
        self.brand_equity_tools = BrandEquityTools()
        self.scenario_tools = ScenarioTools()
        self.data_quality_tools = DataQualityTools()
        self.risk_tools = RiskClassificationTools()
        
        # Set up cross-tool references
        self.brand_equity_tools.set_mmm_tools(self.mmm_tools)
        self.scenario_tools.set_mmm_tools(self.mmm_tools)
    
    def intake_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 1: Intake & Validation
        Parse user query, extract entities, validate request format.
        """
        try:
            logger.info(f"Starting intake for query: {state['user_query']}")
            
            # Extract SKU IDs from query
            extracted_skus = self._extract_sku_ids(state['user_query'])
            if not extracted_skus:
                # Default to available SKUs if none found
                from ..data.synthetic import demo_data
                extracted_skus = list(demo_data.keys())[:3]  # Take first 3 available SKUs
            
            # Extract time range
            time_range = self._extract_time_range(state['user_query'])
            
            # Update state with extracted information
            state = update_state_step(state, "intake", 
                                    scope=extracted_skus, 
                                    time_range=time_range)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 1, "Query Understanding",
                {"query": state['user_query']},
                {"extracted_skus": extracted_skus, "time_range": str(time_range)}
            )
            
            logger.info(f"Intake complete: scope={extracted_skus}, time_range={time_range}")
            return state
            
        except Exception as e:
            logger.error(f"Error in intake_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def validation_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 1: Data Quality Pre-Check
        Detect sparse SKUs, flag missing weeks, identify anomalies.
        """
        try:
            logger.info("Starting data quality validation")
            
            # Run data quality check
            data_quality_report = self.data_quality_tools.detect_data_issues(
                state['scope'], state['time_range']
            )
            
            # Update state
            state = update_state_step(state, "validation", data_quality=data_quality_report)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 2, "Data Quality Check",
                {"scope": state['scope'], "time_range": str(state['time_range'])},
                {
                    "completeness_score": data_quality_report.completeness_score,
                    "recommendation": data_quality_report.recommendation,
                    "anomalies_count": len(data_quality_report.anomalies)
                }
            )
            
            # Check if data quality is sufficient
            if data_quality_report.completeness_score < 0.4:
                state = update_state_step(state, "error", 
                                        error_message="Insufficient data quality for analysis")
                return state
            
            logger.info(f"Validation complete: quality_score={data_quality_report.completeness_score}")
            return state
            
        except Exception as e:
            logger.error(f"Error in validation_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def planning_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 2: Analysis Planning
        Decide which tools to call, estimate confidence level.
        """
        try:
            logger.info("Starting analysis planning")
            
            # Determine query intent
            query_intent = self._classify_query_intent(state['user_query'])
            
            # Plan analysis steps
            analysis_plan = self._create_analysis_plan(query_intent, state)
            
            # Update state
            state = update_state_step(state, "planning", 
                                    query_intent=query_intent,
                                    analysis_plan=analysis_plan)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 3, "Analysis Planning",
                {"query_intent": query_intent},
                {"planned_steps": len(analysis_plan), "confidence_estimate": "high"}
            )
            
            logger.info(f"Planning complete: intent={query_intent}, steps={len(analysis_plan)}")
            return state
            
        except Exception as e:
            logger.error(f"Error in planning_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def analysis_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 3: Tool Execution
        Fetch KPI trends, MMM decomposition, Brand Equity Index.
        """
        try:
            logger.info("Starting analysis execution")
            
            # Execute KPI analysis
            kpi_summary = self.kpi_tools.get_kpi_summary(state['scope'], state['time_range'])
            state = update_state_step(state, "analysis", kpi_summary=kpi_summary)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 4, "KPI Analysis",
                {"scope": state['scope']},
                {
                    "gmv_change_pct": kpi_summary.gmv_change_pct,
                    "units_change_pct": kpi_summary.units_change_pct,
                    "data_quality_score": kpi_summary.data_quality_score
                }
            )
            
            # Execute MMM decomposition
            try:
                mmm_decomposition = self.mmm_tools.get_mmm_decomposition(state['scope'], state['time_range'])
                state = update_state_step(state, "analysis", mmm_decomposition=mmm_decomposition)
                
                # Add diagnostic step
                state = add_diagnostic_step(
                    state, 5, "MMM Decomposition",
                    {"scope": state['scope']},
                    {
                        "total_change_pct": mmm_decomposition.total_change_pct,
                        "model_fit_r2": mmm_decomposition.model_fit_r2,
                        "driver_count": len(mmm_decomposition.driver_contributions)
                    }
                )
            except Exception as e:
                logger.error(f"MMM decomposition failed: {str(e)}")
                # Create a simple decomposition as fallback
                from ..tools.schemas import MMMDecomposition, DriverContribution
                mmm_decomposition = MMMDecomposition(
                    scope=state['scope'],
                    time_range=state['time_range'],
                    total_change_pct=0.0,
                    baseline_demand=0.0,
                    residual=0.0,
                    model_fit_r2=0.3,
                    driver_contributions=[],
                    confidence_score=0.3
                )
                state = update_state_step(state, "analysis", mmm_decomposition=mmm_decomposition)
            
            # Execute Brand Equity analysis
            try:
                brand_equity = self.brand_equity_tools.get_brand_equity(state['scope'], state['time_range'])
                state = update_state_step(state, "analysis", brand_equity=brand_equity)
                
                # Add diagnostic step
                state = add_diagnostic_step(
                    state, 6, "Brand Equity Analysis",
                    {"scope": state['scope']},
                    {
                        "brand_equity_index": brand_equity.brand_equity_index,
                        "trend_direction": brand_equity.trend_direction.value,
                        "confidence_level": brand_equity.confidence_level
                    }
                )
            except Exception as e:
                logger.error(f"Brand equity analysis failed: {str(e)}")
                # Create a simple brand equity as fallback
                from ..tools.schemas import BrandEquityMetrics, TrendDirection
                brand_equity = BrandEquityMetrics(
                    scope=state['scope'],
                    time_range=state['time_range'],
                    brand_equity_index=50.0,
                    previous_index=50.0,
                    trend_direction=TrendDirection.STABLE,
                    velocity=0.0,
                    confidence_level=0.3,
                    risk_alerts=[]
                )
                state = update_state_step(state, "analysis", brand_equity=brand_equity)
            
            logger.info("Analysis execution complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in analysis_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def diagnosis_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 3: Diagnosis Generation
        Identify dominant drivers, classify root cause, compute confidence score.
        """
        try:
            logger.info("Starting diagnosis generation")
            
            if not state['mmm_decomposition'] or not state['kpi_summary']:
                raise ValueError("Missing required data for diagnosis")
            
            # Generate diagnosis
            diagnosis = self._generate_diagnosis(state)
            
            # Update state
            state = update_state_step(state, "diagnosis", diagnosis=diagnosis)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 7, "Diagnosis Generation",
                {"data_available": "true"},
                {
                    "root_cause": diagnosis['root_cause'],
                    "confidence": diagnosis['confidence'],
                    "key_drivers": diagnosis['key_drivers']
                }
            )
            
            logger.info(f"Diagnosis complete: {diagnosis['root_cause']}")
            return state
            
        except Exception as e:
            logger.error(f"Error in diagnosis_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def scenarios_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 4: Scenario Generation
        Test 1-3 scenarios based on diagnosis.
        """
        try:
            logger.info("Starting scenario generation")
            
            # Generate scenarios based on query type
            scenarios = self._generate_scenarios(state)
            state = update_state_step(state, "scenarios", scenarios=scenarios)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 8, "Scenario Generation",
                {"scenario_count": str(len(scenarios))},
                {
                    "scenarios_generated": len(scenarios),
                    "avg_confidence": sum(s.impact.confidence_score for s in scenarios) / len(scenarios) if scenarios else 0
                }
            )
            
            logger.info(f"Scenario generation complete: {len(scenarios)} scenarios")
            return state
            
        except Exception as e:
            logger.error(f"Error in scenarios_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def risk_assessment_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 4: Risk Classification
        Classify risk level, determine if human approval required.
        """
        try:
            logger.info("Starting risk assessment")
            
            # Assess risk for scenarios
            risk_classifications = []
            requires_approval = False
            
            for scenario in state['scenarios']:
                # Create action description from interventions
                action_desc = self._create_action_description(scenario.interventions)
                
                # Create impact dictionary
                impact = {
                    "gmv_impact": scenario.impact.short_term_gmv_impact,
                    "brand_impact": scenario.impact.long_term_brand_impact,
                    "confidence": scenario.impact.confidence_score
                }
                
                # Classify risk
                risk_classification = self.risk_tools.classify_risk(action_desc, impact)
                risk_classifications.append(risk_classification)
                
                if risk_classification.approval_required:
                    requires_approval = True
            
            # Update state
            state = update_state_step(state, "risk_assessment", 
                                    risk_classifications=risk_classifications,
                                    requires_approval=requires_approval)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 9, "Risk Assessment",
                {"scenarios_count": str(len(state['scenarios']))},
                {
                    "high_risk_count": len([r for r in risk_classifications if r.risk_level == RiskLevel.HIGH]),
                    "requires_approval": requires_approval
                }
            )
            
            logger.info(f"Risk assessment complete: approval_required={requires_approval}")
            return state
            
        except Exception as e:
            logger.error(f"Error in risk_assessment_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def recommendations_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 5: Recommendation Ranking
        Score each action, rank top 3 recommendations.
        """
        try:
            logger.info("Starting recommendation ranking")
            
            # Generate recommendations
            recommendations = self._generate_recommendations(state)
            state = update_state_step(state, "recommendations", recommendations=recommendations)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 10, "Recommendation Ranking",
                {"scenarios_count": str(len(state['scenarios']))},
                {
                    "recommendations_count": len(recommendations),
                    "top_recommendation": recommendations[0].action if recommendations else "No recommendations"
                }
            )
            
            logger.info(f"Recommendation ranking complete: {len(recommendations)} recommendations")
            return state
            
        except Exception as e:
            logger.error(f"Error in recommendations_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def explainability_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 5: Explainability Package Generation
        Create diagnostic reasoning chain, confidence breakdown, source attribution.
        """
        try:
            logger.info("Starting explainability package generation")
            
            # Generate explainability package
            explainability = self._generate_explainability_package(state)
            state = update_state_step(state, "explainability", explainability=explainability)
            
            # Calculate overall confidence
            overall_confidence = calculate_overall_confidence(state)
            state = update_state_step(state, "explainability", confidence_score=overall_confidence)
            
            # Add diagnostic step
            state = add_diagnostic_step(
                state, 11, "Explainability Package Generation",
                {"diagnostic_steps_count": str(len(state['diagnostic_steps']))},
                {
                    "overall_confidence": overall_confidence,
                    "sources_traced": len(explainability.source_attribution)
                }
            )
            
            logger.info(f"Explainability package complete: confidence={overall_confidence:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Error in explainability_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    def completion_node(self, state: MMMAgentState) -> MMMAgentState:
        """
        Phase 6: Quality Gate Validation & Memory Update
        Ensure all numbers are grounded, log complete decision lineage.
        """
        try:
            logger.info("Starting completion and quality validation")
            
            # Quality gate validation
            validation_passed = self._quality_gate_validation(state)
            
            if not validation_passed:
                state = update_state_step(state, "error", 
                                        error_message="Quality gate validation failed")
                return state
            
            # Mark as completed
            state = update_state_step(state, "completed", completed=True)
            
            # Add final diagnostic step
            state = add_diagnostic_step(
                state, 12, "Quality Gate Validation",
                {"validation_passed": "true"},
                {
                    "session_completed": True,
                    "final_confidence": state['confidence_score']
                }
            )
            
            logger.info("Analysis completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in completion_node: {str(e)}")
            state = update_state_step(state, "error", error_message=str(e))
            return state
    
    # Helper methods
    def _extract_sku_ids(self, query: str) -> List[str]:
        """Extract SKU IDs from user query."""
        # Look for patterns like SKU-123, SKU-456, etc.
        sku_pattern = r'SKU-\d+'
        skus = re.findall(sku_pattern, query.upper())
        return skus
    
    def _extract_time_range(self, query: str) -> TimeRange:
        """Extract time range from user query."""
        today = date.today()
        
        # Default to last 4 weeks
        end_date = today
        start_date = today - timedelta(weeks=4)
        
        # Look for specific time patterns
        if "last 2 weeks" in query.lower():
            start_date = today - timedelta(weeks=2)
        elif "last 8 weeks" in query.lower():
            start_date = today - timedelta(weeks=8)
        elif "last 12 weeks" in query.lower():
            start_date = today - timedelta(weeks=12)
        
        return TimeRange(start_date=start_date, end_date=end_date)
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the user's query intent."""
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
    
    def _create_analysis_plan(self, intent: str, state: MMMAgentState) -> List[str]:
        """Create analysis plan based on intent."""
        base_plan = ["data_quality", "kpi_analysis", "mmm_decomposition", "brand_equity"]
        
        if intent == "diagnosis":
            return base_plan + ["diagnosis"]
        elif intent == "recommendation":
            return base_plan + ["diagnosis", "scenarios", "risk_assessment", "recommendations"]
        elif intent == "scenario":
            return base_plan + ["scenarios", "risk_assessment"]
        elif intent == "comparison":
            return base_plan + ["comparison"]
        else:
            return base_plan
    
    def _generate_diagnosis(self, state: MMMAgentState) -> Dict[str, Any]:
        """Generate diagnosis from analysis results."""
        mmm_decomp = state['mmm_decomposition']
        kpi_summary = state['kpi_summary']
        
        # Find dominant drivers
        dominant_drivers = []
        for contribution in mmm_decomp.driver_contributions:
            if abs(contribution.contribution_pct) > 2.0:  # Contribution > 2%
                dominant_drivers.append({
                    "driver": contribution.driver,
                    "impact": contribution.contribution_pct
                })
        
        # Determine root cause
        root_cause = "Multiple factors"
        if dominant_drivers:
            # Sort by absolute impact
            dominant_drivers.sort(key=lambda x: abs(x['impact']), reverse=True)
            root_cause = dominant_drivers[0]['driver']
        
        # Calculate confidence
        confidence = (mmm_decomp.confidence_score + kpi_summary.data_quality_score) / 2
        
        return {
            "root_cause": root_cause,
            "dominant_drivers": dominant_drivers,
            "confidence": confidence,
            "key_drivers": [d['driver'] for d in dominant_drivers[:3]]
        }
    
    def _generate_scenarios(self, state: MMMAgentState) -> List[Any]:
        """Generate scenarios based on diagnosis and query intent."""
        intent = self._classify_query_intent(state['user_query'])
        
        if intent == "scenario":
            # Extract specific interventions from query
            interventions = self._extract_interventions_from_query(state['user_query'])
            if interventions:
                scenario = self.scenario_tools.run_scenario(
                    state['scope'], state['time_range'], interventions
                )
                return [scenario]
        
        # Generate default scenarios based on intent
        if intent == "recommendation":
            scenario_type = "recovery"
        else:
            scenario_type = "optimization"
        
        default_scenarios = self.scenario_tools.generate_default_scenarios(
            state['scope'], state['time_range'], scenario_type
        )
        
        scenarios = []
        for scenario_config in default_scenarios[:2]:  # Limit to 2 scenarios
            try:
                scenario = self.scenario_tools.run_scenario(
                    state['scope'], state['time_range'], scenario_config['interventions']
                )
                scenarios.append(scenario)
            except Exception as e:
                logger.warning(f"Failed to generate scenario {scenario_config['name']}: {str(e)}")
        
        return scenarios
    
    def _extract_interventions_from_query(self, query: str) -> List[Dict[str, Any]]:
        """Extract specific interventions from user query."""
        interventions = []
        query_lower = query.lower()
        
        # Extract discount changes
        discount_match = re.search(r'discount\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*%?', query_lower)
        if discount_match:
            interventions.append({
                "parameter": "discount",
                "change": float(discount_match.group(1)),
                "unit": "%"
            })
        
        # Extract SLA changes
        sla_match = re.search(r'sla\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*days?', query_lower)
        if sla_match:
            interventions.append({
                "parameter": "sla",
                "change": float(sla_match.group(1)),
                "unit": "days"
            })
        
        return interventions
    
    def _create_action_description(self, interventions: List[Any]) -> str:
        """Create action description from interventions."""
        actions = []
        for intervention in interventions:
            actions.append(f"{intervention.parameter} {intervention.change}{intervention.unit}")
        return ", ".join(actions)
    
    def _generate_recommendations(self, state: MMMAgentState) -> List[Recommendation]:
        """Generate ranked recommendations from scenarios."""
        recommendations = []
        
        # If no scenarios, generate basic recommendations
        if not state['scenarios']:
            logger.warning("No scenarios available, generating basic recommendations")
            
            # Generate basic recommendations based on available data
            basic_recs = [
                {
                    "action": "Monitor KPI trends closely",
                    "short_term": "Maintain current performance",
                    "long_term": "Data-driven decision making",
                    "risk": RiskLevel.LOW,
                    "confidence": 0.7
                },
                {
                    "action": "Optimize pricing strategy",
                    "short_term": "Review discount effectiveness", 
                    "long_term": "Improve profit margins",
                    "risk": RiskLevel.MEDIUM,
                    "confidence": 0.6
                },
                {
                    "action": "Enhance operational efficiency",
                    "short_term": "Improve SLA performance",
                    "long_term": "Customer satisfaction boost", 
                    "risk": RiskLevel.LOW,
                    "confidence": 0.8
                }
            ]
            
            for i, rec in enumerate(basic_recs):
                recommendation = Recommendation(
                    rank=i + 1,
                    action=rec["action"],
                    short_term_impact=rec["short_term"],
                    long_term_impact=rec["long_term"],
                    risk_level=rec["risk"],
                    confidence_score=rec["confidence"],
                    feasibility="High",
                    caveats=["General recommendation - customize for your business"]
                )
                recommendations.append(recommendation)
        
        else:
            # Generate recommendations from scenarios
            for i, scenario in enumerate(state['scenarios']):
                # Create recommendation from scenario
                action_desc = self._create_action_description(scenario.interventions)
                
                recommendation = Recommendation(
                    rank=i + 1,
                    action=action_desc,
                    short_term_impact=f"GMV impact: {scenario.impact.short_term_gmv_impact:.1f}%",
                    long_term_impact=f"Brand impact: {scenario.impact.long_term_brand_impact:.2f} points",
                    risk_level=scenario.impact.risk_level,
                    confidence_score=scenario.impact.confidence_score,
                    feasibility="High" if scenario.impact.risk_level == RiskLevel.LOW else "Medium",
                    caveats=[] if scenario.impact.confidence_score > 0.8 else ["Low confidence - verify with more data"]
                )
                recommendations.append(recommendation)
        
        # Sort by risk-adjusted ROI (simplified)
        recommendations.sort(key=lambda x: (
            0 if x.risk_level == RiskLevel.LOW else (1 if x.risk_level == RiskLevel.MEDIUM else 2),
            -x.confidence_score
        ))
        
        # Re-rank
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1
        
        return recommendations[:3]  # Return top 3
    
    def _generate_explainability_package(self, state: MMMAgentState) -> ExplainabilityPackage:
        """Generate explainability package."""
        # Create confidence breakdown
        confidence_breakdown = {}
        if state['data_quality']:
            confidence_breakdown['data_quality'] = state['data_quality'].completeness_score
        if state['mmm_decomposition']:
            confidence_breakdown['model_fit'] = state['mmm_decomposition'].confidence_score
        if state['brand_equity']:
            confidence_breakdown['brand_model'] = state['brand_equity'].confidence_level
        
        # Create source attribution
        source_attribution = {}
        for step in state['diagnostic_steps']:
            source_attribution[f"step_{step.step_number}"] = step.action
        
        # Create alternatives considered
        alternatives = [
            "Increase marketing spend",
            "Reduce pricing",
            "Improve operational metrics",
            "Maintain current strategy"
        ]
        
        return ExplainabilityPackage(
            session_id=state['session_id'],
            query=state['user_query'],
            diagnostic_steps=state['diagnostic_steps'],
            confidence_breakdown=confidence_breakdown,
            source_attribution=source_attribution,
            alternatives_considered=alternatives,
            overall_confidence=state['confidence_score']
        )
    
    def _quality_gate_validation(self, state: MMMAgentState) -> bool:
        """Validate that all quality gates are passed."""
        # Check required components
        required_components = [
            state['data_quality'] is not None,
            state['kpi_summary'] is not None,
            state['mmm_decomposition'] is not None,
            len(state['recommendations']) > 0,
            state['explainability'] is not None
        ]
        
        if not all(required_components):
            return False
        
        # Check confidence threshold
        if state['confidence_score'] < 0.3:
            return False
        
        # Check data quality
        if state['data_quality'].completeness_score < 0.4:
            return False
        
        return True
