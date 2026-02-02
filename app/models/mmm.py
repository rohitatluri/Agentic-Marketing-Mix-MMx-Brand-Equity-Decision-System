"""MMM computation models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from ..tools.schemas import (
    MMMDecomposition, DriverContribution, TimeRange, 
    ScenarioResult, Intervention, ScenarioImpact, RiskLevel
)


class MarketingMixModel:
    """Marketing Mix Model for attributing sales changes to drivers."""
    
    def __init__(self):
        """Initialize MMM with default parameters."""
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.coefficients = {}
        self.intercept = 0
        
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for MMM regression."""
        features = pd.DataFrame()
        
        # Price/Discount features
        features['discount_pct'] = data['discount_pct']
        features['price_change'] = (data['current_price'] - data['base_price']) / data['base_price']
        
        # Service features
        features['sla_days'] = data['sla_days']
        features['sla_success_rate'] = data['sla_success_rate']
        
        # Supply features
        features['procurement_sla_days'] = data['procurement_sla_days']
        features['stockout_rate'] = data['stockout_rate']
        
        # Marketing features
        features['marketing_spend'] = data['marketing_spend']
        features['marketing_spend_log'] = np.log1p(data['marketing_spend'])
        
        # Competition
        features['competitor_price_index'] = data['competitor_price_index']
        
        # Lagged demand (previous week)
        features['lagged_demand'] = data['units'].shift(1).fillna(data['units'].mean())
        
        # Time features
        features['week_number'] = data['week_number']
        features['week_sin'] = np.sin(2 * np.pi * data['week_number'] / 52)
        features['week_cos'] = np.cos(2 * np.pi * data['week_number'] / 52)
        
        return features
    
    def fit(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Fit the MMM model to historical data.
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            Dictionary with model fit metrics
        """
        if len(data) < 8:
            raise ValueError("Need at least 8 weeks of data for MMM")
        
        # Prepare features and target
        features = self._prepare_features(data)
        target = data['units']
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(features_scaled, target)
        self.is_fitted = True
        
        # Store coefficients
        self.coefficients = dict(zip(self.feature_names, self.model.coef_))
        self.intercept = self.model.intercept_
        
        # Calculate metrics
        predictions = self.model.predict(features_scaled)
        r2 = r2_score(target, predictions)
        
        return {
            'r2_score': r2,
            'data_points': len(data),
            'feature_count': len(self.feature_names)
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def decompose_changes(self, data: pd.DataFrame, time_range: TimeRange) -> MMMDecomposition:
        """
        Decompose sales changes into driver contributions.
        
        Args:
            data: DataFrame with data for the time range
            time_range: Analysis time range
            
        Returns:
            MMMDecomposition with driver contributions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decomposition")
        
        # Split data into current and previous periods
        mid_point = len(data) // 2
        previous_data = data.iloc[:mid_point]
        current_data = data.iloc[mid_point:]
        
        # Calculate average units for each period
        prev_units = previous_data['units'].mean()
        curr_units = current_data['units'].mean()
        
        # Calculate total change
        total_change = curr_units - prev_units
        total_change_pct = (total_change / prev_units) * 100 if prev_units > 0 else 0
        
        # Prepare features for both periods
        prev_features = self._prepare_features(previous_data)
        curr_features = self._prepare_features(current_data)
        
        # Scale features
        prev_features_scaled = self.scaler.transform(prev_features)
        curr_features_scaled = self.scaler.transform(curr_features)
        
        # Calculate average feature values
        prev_avg = prev_features_scaled.mean(axis=0)
        curr_avg = curr_features_scaled.mean(axis=0)
        
        # Calculate contributions for each driver
        contributions = []
        driver_mapping = {
            'discount_pct': 'Price/Discount',
            'price_change': 'Price/Discount',
            'sla_days': 'SLA',
            'sla_success_rate': 'SLA',
            'procurement_sla_days': 'Procurement',
            'stockout_rate': 'Procurement',
            'marketing_spend': 'Marketing',
            'marketing_spend_log': 'Marketing',
            'competitor_price_index': 'Competition',
            'lagged_demand': 'Seasonal/Trend',
            'week_number': 'Seasonal/Trend',
            'week_sin': 'Seasonal/Trend',
            'week_cos': 'Seasonal/Trend'
        }
        
        # Group contributions by driver category
        driver_contributions = {}
        for i, feature in enumerate(self.feature_names):
            if feature in self.coefficients:
                driver_category = driver_mapping.get(feature, 'Other')
                feature_change = (curr_avg[i] - prev_avg[i]) * self.coefficients[feature]
                
                if driver_category not in driver_contributions:
                    driver_contributions[driver_category] = 0
                driver_contributions[driver_category] += feature_change
        
        # Convert to contribution objects
        baseline_demand = prev_units
        explained_change = 0
        
        for driver, contribution in driver_contributions.items():
            contribution_pct = (contribution / prev_units) * 100 if prev_units > 0 else 0
            impact_amount = contribution * (curr_units / (curr_units + baseline_demand)) if curr_units > 0 else 0
            
            contributions.append(DriverContribution(
                driver=driver,
                contribution_pct=contribution_pct,
                impact_amount=impact_amount,
                confidence_interval={
                    "lower": contribution_pct * 0.8,
                    "upper": contribution_pct * 1.2
                }
            ))
            explained_change += contribution
        
        # Calculate residual (unexplained variation)
        residual = total_change - explained_change
        residual_pct = (residual / prev_units) * 100 if prev_units > 0 else 0
        
        # Add residual as a contribution
        contributions.append(DriverContribution(
            driver="Unexplained/Residual",
            contribution_pct=residual_pct,
            impact_amount=residual,
            confidence_interval={"lower": residual_pct * 0.7, "upper": residual_pct * 1.3}
        ))
        
        # Calculate confidence score based on model fit and data quality
        model_fit_score = min(1.0, self.model.score(self.scaler.transform(self._prepare_features(data)), data['units']))
        data_quality_score = min(1.0, len(data) / 16)  # More data = higher confidence
        confidence_score = (model_fit_score * 0.6 + data_quality_score * 0.4)
        
        return MMMDecomposition(
            scope=data['sku_id'].unique().tolist(),
            time_range=time_range,
            total_change_pct=total_change_pct,
            baseline_demand=baseline_demand,
            residual=residual,
            model_fit_r2=model_fit_score,
            driver_contributions=contributions,
            confidence_score=confidence_score
        )
    
    def simulate_scenario(self, 
                         base_data: pd.DataFrame, 
                         interventions: List[Intervention],
                         time_range: TimeRange) -> ScenarioResult:
        """
        Simulate the impact of interventions.
        
        Args:
            base_data: Base data for simulation
            interventions: List of interventions to apply
            time_range: Analysis time range
            
        Returns:
            ScenarioResult with impact analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        # Create modified data by applying interventions
        modified_data = base_data.copy()
        
        for intervention in interventions:
            if intervention.parameter == "discount":
                # Apply discount change
                if intervention.unit == "%":
                    modified_data['discount_pct'] += intervention.change / 100
                    modified_data['current_price'] = modified_data['base_price'] * (1 - modified_data['discount_pct'])
            
            elif intervention.parameter == "sla":
                # Apply SLA change
                if intervention.unit == "days":
                    modified_data['sla_days'] += intervention.change
                    modified_data['sla_success_rate'] = np.clip(
                        modified_data['sla_success_rate'] - intervention.change * 0.02, 0.7, 0.99
                    )
            
            elif intervention.parameter == "procurement":
                # Apply procurement SLA change
                if intervention.unit == "days":
                    modified_data['procurement_sla_days'] += intervention.change
                    modified_data['stockout_rate'] = np.clip(
                        modified_data['stockout_rate'] - intervention.change * 0.01, 0, 0.3
                    )
            
            elif intervention.parameter == "marketing":
                # Apply marketing spend change
                if intervention.unit == "%":
                    modified_data['marketing_spend'] *= (1 + intervention.change / 100)
        
        # Predict with base and modified data
        base_predictions = self.predict(base_data)
        modified_predictions = self.predict(modified_data)
        
        # Calculate impacts
        base_avg_units = base_predictions.mean()
        modified_avg_units = modified_predictions.mean()
        
        # Calculate GMV impact (using average price)
        base_avg_price = base_data['current_price'].mean()
        modified_avg_price = modified_data['current_price'].mean()
        
        base_gmv = base_avg_units * base_avg_price
        modified_gmv = modified_avg_units * modified_avg_price
        
        short_term_gmv_impact = ((modified_gmv - base_gmv) / base_gmv) * 100 if base_gmv > 0 else 0
        
        # Estimate brand impact (simplified heuristic)
        brand_impact = 0
        for intervention in interventions:
            if intervention.parameter == "discount" and intervention.change > 0:
                brand_impact -= intervention.change * 0.1  # Discounts hurt brand
            elif intervention.parameter == "sla" and intervention.change < 0:
                brand_impact += abs(intervention.change) * 0.2  # Better SLA helps brand
            elif intervention.parameter == "procurement" and intervention.change < 0:
                brand_impact += abs(intervention.change) * 0.15  # Better procurement helps brand
        
        # Calculate net ROI (simplified)
        net_roi = short_term_gmv_impact + (brand_impact * 3)  # Weight brand impact more heavily
        
        # Classify risk
        risk_level = self._classify_scenario_risk(interventions, short_term_gmv_impact, brand_impact)
        
        # Calculate confidence
        confidence = min(0.9, self.model.score(
            self.scaler.transform(self._prepare_features(base_data)), 
            base_data['units']
        ))
        
        # Create reasoning
        reasoning = f"Applied {len(interventions)} interventions: "
        reasoning += ", ".join([f"{intervention.parameter} {intervention.change}{intervention.unit}" 
                              for intervention in interventions])
        reasoning += f". Expected short-term GMV impact: {short_term_gmv_impact:.1f}%, Brand impact: {brand_impact:.2f} points"
        
        return ScenarioResult(
            interventions=interventions,
            impact=ScenarioImpact(
                short_term_gmv_impact=short_term_gmv_impact,
                long_term_brand_impact=brand_impact,
                net_roi=net_roi,
                risk_level=risk_level,
                confidence_score=confidence,
                uncertainty_band={
                    "lower": short_term_gmv_impact * 0.7,
                    "upper": short_term_gmv_impact * 1.3
                }
            ),
            reasoning=reasoning
        )
    
    def _classify_scenario_risk(self, interventions: List[Intervention], gmv_impact: float, brand_impact: float) -> RiskLevel:
        """Classify the risk level of a scenario."""
        risk_score = 0
        
        for intervention in interventions:
            if intervention.parameter == "discount":
                if intervention.change > 10:  # Discount increase > 10%
                    risk_score += 3
                elif intervention.change > 5:
                    risk_score += 2
                else:
                    risk_score += 1
            
            elif intervention.parameter == "sla" and intervention.change > 2:
                risk_score += 2  # Significant SLA degradation
            
            elif intervention.parameter == "marketing" and intervention.change > 50:
                risk_score += 2  # Large marketing spend increase
        
        # Consider brand impact
        if brand_impact < -0.5:
            risk_score += 3
        elif brand_impact < -0.2:
            risk_score += 2
        
        # Classify based on score
        if risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
