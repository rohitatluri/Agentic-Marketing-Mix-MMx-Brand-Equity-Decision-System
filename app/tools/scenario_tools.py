"""Scenario simulation tools."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union
import logging

from ..data.synthetic import get_demo_data
from ..models.mmm import MarketingMixModel
from ..tools.schemas import ScenarioResult, Intervention, TimeRange

logger = logging.getLogger(__name__)


class ScenarioTools:
    """Tools for scenario simulation and what-if analysis."""
    
    def __init__(self):
        """Initialize scenario tools."""
        self.demo_data = get_demo_data()
        self.mmm_tools = None  # Will be set to avoid circular import
        
    def set_mmm_tools(self, mmm_tools):
        """Set MMM tools reference (to avoid circular import)."""
        self.mmm_tools = mmm_tools
        
    def run_scenario(self, 
                    scope: List[str], 
                    time_range: TimeRange, 
                    interventions: List[Dict[str, Union[str, float]]]) -> ScenarioResult:
        """
        Simulate what-if scenarios with specified interventions.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            interventions: List of interventions (e.g., [{"parameter": "discount", "change": -5, "unit": "%"}])
            
        Returns:
            ScenarioResult with impact analysis
        """
        # Validate inputs
        self._validate_inputs(scope, time_range, interventions)
        
        # Convert interventions to Intervention objects
        intervention_objects = []
        for intervention_dict in interventions:
            intervention_objects.append(Intervention(**intervention_dict))
        
        # Collect data for all SKUs in scope
        all_data = []
        for sku_id in scope:
            if sku_id in self.demo_data:
                sku_data = self.demo_data[sku_id].copy()
                # Filter by time range - convert both to datetime for comparison
                sku_data['week_date'] = pd.to_datetime(sku_data['week_date'])
                start_date = pd.to_datetime(time_range.start_date)
                end_date = pd.to_datetime(time_range.end_date)
                mask = (sku_data['week_date'] >= start_date) & \
                       (sku_data['week_date'] <= end_date)
                filtered_data = sku_data[mask]
                all_data.append(filtered_data)
        
        if not all_data:
            raise ValueError(f"No data found for SKUs: {scope}")
        
        # Combine all SKU data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Check if we have enough data
        if len(combined_data) < 3:
            raise ValueError(f"Insufficient data for scenario analysis. Need at least 3 weeks, got {len(combined_data)}")
        
        # If we have less than 6 weeks, use a simpler approach
        if len(combined_data) < 6:
            logger.warning(f"Using simplified scenario analysis with {len(combined_data)} weeks (minimum 6 recommended)")
        
        # Get MMM model
        if self.mmm_tools is None:
            # Create a simple MMM model
            mmm_model = MarketingMixModel()
            mmm_model.fit(combined_data)
        else:
            # Get MMM model from tools
            model_key = tuple(sorted(scope))
            if model_key not in self.mmm_tools.mmm_models:
                self.mmm_tools.get_mmm_decomposition(scope, time_range)
            mmm_model = self.mmm_tools.mmm_models[model_key]
        
        # Run scenario simulation
        scenario_result = mmm_model.simulate_scenario(combined_data, intervention_objects, time_range)
        
        return scenario_result
    
    def _validate_inputs(self, scope: List[str], time_range: TimeRange, interventions: List[Dict]):
        """Validate tool inputs."""
        if not scope:
            raise ValueError("Scope cannot be empty")
        
        if len(scope) > 100:
            raise ValueError("Scope cannot exceed 100 SKUs")
        
        if time_range.end_date <= time_range.start_date:
            raise ValueError("End date must be after start date")
        
        if not interventions:
            raise ValueError("At least one intervention must be specified")
        
        if len(interventions) > 3:
            raise ValueError("Cannot simulate more than 3 interventions at once")
        
        # Validate each intervention
        valid_parameters = {"discount", "sla", "procurement", "marketing"}
        valid_units = {"%", "days", "$"}
        
        for intervention in interventions:
            if "parameter" not in intervention:
                raise ValueError("Each intervention must specify a parameter")
            
            if intervention["parameter"] not in valid_parameters:
                raise ValueError(f"Invalid parameter: {intervention['parameter']}. Valid options: {valid_parameters}")
            
            if "change" not in intervention:
                raise ValueError("Each intervention must specify a change")
            
            if "unit" not in intervention:
                raise ValueError("Each intervention must specify a unit")
            
            if intervention["unit"] not in valid_units:
                raise ValueError(f"Invalid unit: {intervention['unit']}. Valid options: {valid_units}")
            
            # Check for unrealistic interventions
            if intervention["parameter"] == "discount" and abs(intervention["change"]) > 50:
                raise ValueError("Discount changes cannot exceed 50%")
            
            if intervention["parameter"] == "sla" and abs(intervention["change"]) > 10:
                raise ValueError("SLA changes cannot exceed 10 days")
            
            if intervention["parameter"] == "marketing" and abs(intervention["change"]) > 200:
                raise ValueError("Marketing spend changes cannot exceed 200%")
    
    def generate_default_scenarios(self, 
                                 scope: List[str], 
                                 time_range: TimeRange,
                                 scenario_type: str = "recovery") -> List[Dict[str, Union[str, float]]]:
        """
        Generate default scenarios based on analysis type.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            scenario_type: Type of scenarios to generate ("recovery", "optimization", "growth")
            
        Returns:
            List of scenario configurations
        """
        scenarios = []
        
        if scenario_type == "recovery":
            # Recovery scenarios - focus on fixing issues
            scenarios = [
                {
                    "name": "Improve SLA",
                    "interventions": [
                        {"parameter": "sla", "change": -1, "unit": "days"}
                    ]
                },
                {
                    "name": "Reduce Stockouts",
                    "interventions": [
                        {"parameter": "procurement", "change": -2, "unit": "days"}
                    ]
                },
                {
                    "name": "Moderate Discount",
                    "interventions": [
                        {"parameter": "discount", "change": 5, "unit": "%"}
                    ]
                }
            ]
        
        elif scenario_type == "optimization":
            # Optimization scenarios - balance growth and brand health
            scenarios = [
                {
                    "name": "SLA + Marketing",
                    "interventions": [
                        {"parameter": "sla", "change": -1, "unit": "days"},
                        {"parameter": "marketing", "change": 20, "unit": "%"}
                    ]
                },
                {
                    "name": "Price Optimization",
                    "interventions": [
                        {"parameter": "discount", "change": -3, "unit": "%"},
                        {"parameter": "marketing", "change": 30, "unit": "%"}
                    ]
                },
                {
                    "name": "Supply Chain Focus",
                    "interventions": [
                        {"parameter": "procurement", "change": -1, "unit": "days"},
                        {"parameter": "sla", "change": -0.5, "unit": "days"}
                    ]
                }
            ]
        
        elif scenario_type == "growth":
            # Growth scenarios - aggressive expansion
            scenarios = [
                {
                    "name": "Aggressive Marketing",
                    "interventions": [
                        {"parameter": "marketing", "change": 50, "unit": "%"},
                        {"parameter": "discount", "change": 10, "unit": "%"}
                    ]
                },
                {
                    "name": "Premium Strategy",
                    "interventions": [
                        {"parameter": "discount", "change": -5, "unit": "%"},
                        {"parameter": "sla", "change": -2, "unit": "days"}
                    ]
                },
                {
                    "name": "Balanced Growth",
                    "interventions": [
                        {"parameter": "marketing", "change": 25, "unit": "%"},
                        {"parameter": "sla", "change": -1, "unit": "days"}
                    ]
                }
            ]
        
        return scenarios
    
    def compare_scenarios(self, 
                         scope: List[str], 
                         time_range: TimeRange,
                         scenarios: List[Dict[str, Union[str, float]]]) -> Dict[str, ScenarioResult]:
        """
        Compare multiple scenarios side by side.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            scenarios: List of scenario configurations
            
        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get("name", f"Scenario_{len(results)+1}")
            interventions = scenario.get("interventions", [])
            
            try:
                result = self.run_scenario(scope, time_range, interventions)
                results[scenario_name] = result
            except Exception as e:
                logger.error(f"Error running scenario {scenario_name}: {str(e)}")
                # Create a dummy result for failed scenarios
                from ..tools.schemas import ScenarioImpact, RiskLevel
                dummy_result = ScenarioResult(
                    interventions=[Intervention(**intervention) for intervention in interventions],
                    impact=ScenarioImpact(
                        short_term_gmv_impact=0.0,
                        long_term_brand_impact=0.0,
                        net_roi=0.0,
                        risk_level=RiskLevel.HIGH,
                        confidence_score=0.0
                    ),
                    reasoning=f"Scenario failed: {str(e)}"
                )
                results[scenario_name] = dummy_result
        
        return results
    
    def get_scenario_recommendations(self, 
                                   scenario_results: Dict[str, ScenarioResult],
                                   objective: str = "balanced") -> List[str]:
        """
        Get recommendations based on scenario comparison.
        
        Args:
            scenario_results: Results from scenario comparison
            objective: Optimization objective ("growth", "brand", "balanced")
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not scenario_results:
            return ["No valid scenarios to compare"]
        
        # Sort scenarios by objective
        if objective == "growth":
            # Prioritize short-term GMV impact
            sorted_scenarios = sorted(
                scenario_results.items(),
                key=lambda x: x[1].impact.short_term_gmv_impact,
                reverse=True
            )
        elif objective == "brand":
            # Prioritize brand impact
            sorted_scenarios = sorted(
                scenario_results.items(),
                key=lambda x: x[1].impact.long_term_brand_impact,
                reverse=True
            )
        else:  # balanced
            # Prioritize net ROI
            sorted_scenarios = sorted(
                scenario_results.items(),
                key=lambda x: x[1].impact.net_roi,
                reverse=True
            )
        
        # Generate recommendations
        if sorted_scenarios:
            best_scenario = sorted_scenarios[0]
            recommendations.append(
                f"RECOMMENDED: {best_scenario[0]} - "
                f"Expected GMV impact: {best_scenario[1].impact.short_term_gmv_impact:.1f}%, "
                f"Brand impact: {best_scenario[1].impact.long_term_brand_impact:.2f} points, "
                f"Risk: {best_scenario[1].impact.risk_level.value}"
            )
            
            # Add second best as alternative
            if len(sorted_scenarios) > 1:
                second_best = sorted_scenarios[1]
                recommendations.append(
                    f"ALTERNATIVE: {second_best[0]} - "
                    f"Expected GMV impact: {second_best[1].impact.short_term_gmv_impact:.1f}%, "
                    f"Brand impact: {second_best[1].impact.long_term_brand_impact:.2f} points"
                )
        
        # Add risk warnings
        high_risk_scenarios = [
            name for name, result in scenario_results.items() 
            if result.impact.risk_level == RiskLevel.HIGH
        ]
        
        if high_risk_scenarios:
            recommendations.append(
                f"⚠️ HIGH RISK: {', '.join(high_risk_scenarios)} - "
                "These scenarios may damage brand equity and require careful consideration"
            )
        
        return recommendations
