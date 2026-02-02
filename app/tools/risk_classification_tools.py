"""Risk classification tools for HITL support."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union
import logging

from ..tools.schemas import RiskClassification, RiskLevel

logger = logging.getLogger(__name__)


class RiskClassificationTools:
    """Tools for classifying risk levels and determining HITL requirements."""
    
    def __init__(self):
        """Initialize risk classification tools."""
        # Risk thresholds (configurable)
        self.risk_thresholds = {
            "discount_change_high": 10.0,  # % change
            "discount_change_medium": 5.0,  # % change
            "budget_change_high": 50000.0,  # $
            "budget_change_medium": 20000.0,  # $
            "brand_impact_high": -0.5,  # points
            "brand_impact_medium": -0.2,  # points
            "confidence_low": 0.7,  # threshold
            "confidence_medium": 0.8  # threshold
        }
        
    def classify_risk(self, action: str, impact: Optional[Dict[str, float]] = None) -> RiskClassification:
        """
        Determine if action requires human approval based on risk classification.
        
        Args:
            action: Proposed intervention (e.g., "increase discount by 20%")
            impact: Expected outcomes (GMV gain, brand damage, confidence)
            
        Returns:
            RiskClassification with risk level and HITL requirements
        """
        # Parse action to extract parameters
        action_params = self._parse_action(action)
        
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        
        # Check discount changes
        if "discount" in action_params:
            discount_change = abs(action_params["discount"])
            if discount_change > self.risk_thresholds["discount_change_high"]:
                risk_score += 3
                risk_factors.append(f"High discount change ({discount_change:.1f}%)")
            elif discount_change > self.risk_thresholds["discount_change_medium"]:
                risk_score += 2
                risk_factors.append(f"Medium discount change ({discount_change:.1f}%)")
            else:
                risk_score += 1
                risk_factors.append(f"Low discount change ({discount_change:.1f}%)")
        
        # Check budget changes (if specified in impact)
        if impact and "budget_change" in impact:
            budget_change = abs(impact["budget_change"])
            if budget_change > self.risk_thresholds["budget_change_high"]:
                risk_score += 3
                risk_factors.append(f"High budget change (${budget_change:,.0f})")
            elif budget_change > self.risk_thresholds["budget_change_medium"]:
                risk_score += 2
                risk_factors.append(f"Medium budget change (${budget_change:,.0f})")
        
        # Check brand impact
        if impact and "brand_impact" in impact:
            brand_impact = impact["brand_impact"]
            if brand_impact < self.risk_thresholds["brand_impact_high"]:
                risk_score += 3
                risk_factors.append(f"High brand damage ({brand_impact:.2f} points)")
            elif brand_impact < self.risk_thresholds["brand_impact_medium"]:
                risk_score += 2
                risk_factors.append(f"Medium brand damage ({brand_impact:.2f} points)")
        
        # Check confidence level
        if impact and "confidence" in impact:
            confidence = impact["confidence"]
            if confidence < self.risk_thresholds["confidence_low"]:
                risk_score += 3
                risk_factors.append(f"Low confidence ({confidence:.1%})")
            elif confidence < self.risk_thresholds["confidence_medium"]:
                risk_score += 1
                risk_factors.append(f"Medium confidence ({confidence:.1%})")
        
        # Check for other high-risk factors
        if "sla" in action_params and action_params["sla"] > 5:  # SLA degradation > 5 days
            risk_score += 2
            risk_factors.append("Significant SLA degradation")
        
        if "marketing" in action_params and action_params["marketing"] > 100:  # Marketing increase > 100%
            risk_score += 2
            risk_factors.append("High marketing spend increase")
        
        # Determine risk level
        if risk_score >= 4:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Determine if approval is required
        approval_required = risk_level == RiskLevel.HIGH
        
        # Generate reasoning
        reasoning = f"Risk score: {risk_score}. Factors: {', '.join(risk_factors)}"
        
        # Generate HITL message if needed
        hitl_message = None
        if approval_required:
            hitl_message = self._generate_hitl_message(action, impact, risk_factors)
        
        # Extract short-term gain and long-term cost
        short_term_gain = impact.get("gmv_impact") if impact else None
        long_term_cost = impact.get("brand_cost") if impact else None
        
        return RiskClassification(
            action=action,
            risk_level=risk_level,
            reasoning=reasoning,
            approval_required=approval_required,
            hitl_message=hitl_message,
            short_term_gain=short_term_gain,
            long_term_cost=long_term_cost
        )
    
    def _parse_action(self, action: str) -> Dict[str, float]:
        """Parse action string to extract parameters and values."""
        params = {}
        action_lower = action.lower()
        
        # Extract discount changes
        import re
        discount_match = re.search(r'discount\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*%?', action_lower)
        if discount_match:
            params["discount"] = float(discount_match.group(1))
        
        # Extract SLA changes
        sla_match = re.search(r'sla\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*days?', action_lower)
        if sla_match:
            params["sla"] = float(sla_match.group(1))
        
        # Extract procurement changes
        proc_match = re.search(r'procurement\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*days?', action_lower)
        if proc_match:
            params["procurement"] = float(proc_match.group(1))
        
        # Extract marketing changes
        marketing_match = re.search(r'marketing\s+(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*%?', action_lower)
        if marketing_match:
            params["marketing"] = float(marketing_match.group(1))
        
        return params
    
    def _generate_hitl_message(self, action: str, impact: Optional[Dict[str, float]], risk_factors: List[str]) -> str:
        """Generate HITL message for high-risk actions."""
        message = "⚠️ HIGH RISK ACTION DETECTED\n\n"
        message += f"Proposed Action: {action}\n\n"
        
        if impact:
            message += "Expected Impact:\n"
            if "gmv_impact" in impact:
                gmv_impact = impact["gmv_impact"]
                if gmv_impact > 0:
                    message += f"✅ Short-term GMV: +${gmv_impact:,.0f}\n"
                else:
                    message += f"❌ Short-term GMV: ${gmv_impact:,.0f}\n"
            
            if "brand_impact" in impact:
                brand_impact = impact["brand_impact"]
                if brand_impact < 0:
                    message += f"❌ Brand Equity: {brand_impact:.2f} points (significant erosion)\n"
                else:
                    message += f"✅ Brand Equity: +{brand_impact:.2f} points\n"
            
            if "net_roi" in impact:
                net_roi = impact["net_roi"]
                if net_roi < 0:
                    message += f"❌ Net 12-week impact: ${net_roi:,.0f}\n"
                else:
                    message += f"✅ Net 12-week impact: +${net_roi:,.0f}\n"
        
        message += f"\nRisk Factors:\n"
        for factor in risk_factors:
            message += f"• {factor}\n"
        
        message += f"\nConfidence: {impact.get('confidence', 0):.0%}\n"
        message += "\nThis decision requires manager approval."
        
        return message
    
    def batch_classify_risks(self, actions: List[str], impacts: Optional[List[Dict[str, float]]] = None) -> List[RiskClassification]:
        """
        Classify risk for multiple actions.
        
        Args:
            actions: List of proposed actions
            impacts: List of impact dictionaries (optional)
            
        Returns:
            List of risk classifications
        """
        if impacts is None:
            impacts = [None] * len(actions)
        
        if len(actions) != len(impacts):
            raise ValueError("Actions and impacts must have the same length")
        
        classifications = []
        for action, impact in zip(actions, impacts):
            classification = self.classify_risk(action, impact)
            classifications.append(classification)
        
        return classifications
    
    def get_risk_summary(self, classifications: List[RiskClassification]) -> Dict[str, Union[int, List[str]]]:
        """
        Get a summary of risk classifications.
        
        Args:
            classifications: List of risk classifications
            
        Returns:
            Dictionary with risk summary
        """
        high_risk_actions = [c.action for c in classifications if c.risk_level == RiskLevel.HIGH]
        medium_risk_actions = [c.action for c in classifications if c.risk_level == RiskLevel.MEDIUM]
        low_risk_actions = [c.action for c in classifications if c.risk_level == RiskLevel.LOW]
        
        approval_required = [c.action for c in classifications if c.approval_required]
        
        return {
            "total_actions": len(classifications),
            "high_risk_count": len(high_risk_actions),
            "medium_risk_count": len(medium_risk_actions),
            "low_risk_count": len(low_risk_actions),
            "approval_required_count": len(approval_required),
            "high_risk_actions": high_risk_actions,
            "approval_required_actions": approval_required
        }
    
    def update_risk_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update risk classification thresholds.
        
        Args:
            new_thresholds: Dictionary of new threshold values
        """
        for key, value in new_thresholds.items():
            if key in self.risk_thresholds:
                self.risk_thresholds[key] = value
                logger.info(f"Updated risk threshold {key} to {value}")
            else:
                logger.warning(f"Unknown risk threshold: {key}")
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current risk classification thresholds."""
        return self.risk_thresholds.copy()
