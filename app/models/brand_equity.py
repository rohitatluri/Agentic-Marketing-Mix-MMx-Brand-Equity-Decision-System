"""Brand index computation models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..tools.schemas import (
    BrandEquityMetrics, TimeRange, TrendDirection
)


class BrandEquityModel:
    """Brand Equity Index computation model for tracking long-term brand health."""
    
    def __init__(self):
        """Initialize Brand Equity model with default parameters."""
        self.is_fitted = False
        self.base_residuals = {}
        self.brand_sensitivity = {
            'sla_consistency': -0.3,
            'procurement_reliability': -0.25,
            'price_volatility': -0.4,
            'discount_intensity': -0.5
        }
        
    def _calculate_residuals(self, data: pd.DataFrame, mmm_model) -> pd.Series:
        """
        Calculate residuals after MMM decomposition.
        
        Args:
            data: DataFrame with sales data
            mmm_model: Fitted MMM model
            
        Returns:
            Series of residuals
        """
        if not mmm_model.is_fitted:
            raise ValueError("MMM model must be fitted first")
        
        # Get predictions from MMM
        predictions = mmm_model.predict(data)
        
        # Calculate residuals (actual - predicted)
        residuals = data['units'] - predictions
        
        return residuals
    
    def _smooth_residuals(self, residuals: pd.Series, window: int = 4) -> pd.Series:
        """
        Apply moving average smoothing to residuals.
        
        Args:
            residuals: Series of residuals
            window: Moving average window size
            
        Returns:
            Smoothed residuals
        """
        return residuals.rolling(window=window, center=True).mean().fillna(residuals)
    
    def _normalize_to_index(self, values: pd.Series, min_val: float = 0, max_val: float = 100) -> pd.Series:
        """
        Normalize values to 0-100 index scale.
        
        Args:
            values: Series of values to normalize
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            
        Returns:
            Normalized series (0-100 scale)
        """
        min_actual = values.min()
        max_actual = values.max()
        
        if max_actual == min_actual:
            return pd.Series([50] * len(values), index=values.index)
        
        # Normalize to 0-100 scale
        normalized = ((values - min_actual) / (max_actual - min_actual)) * (max_val - min_val) + min_val
        return normalized
    
    def _calculate_trend_direction(self, values: pd.Series, window: int = 8) -> TrendDirection:
        """
        Calculate trend direction based on linear regression.
        
        Args:
            values: Series of values
            window: Window size for trend calculation
            
        Returns:
            TrendDirection enum
        """
        if len(values) < window:
            return TrendDirection.STABLE
        
        # Use last 'window' values for trend calculation
        recent_values = values.tail(window).values
        x = np.arange(len(recent_values))
        
        # Calculate slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        
        # Determine trend based on slope and statistical significance
        if p_value < 0.1:  # Statistically significant trend
            if slope > 0.1:  # Positive slope
                return TrendDirection.IMPROVING
            elif slope < -0.1:  # Negative slope
                return TrendDirection.DECLINING
        
        return TrendDirection.STABLE
    
    def _calculate_velocity(self, values: pd.Series, window: int = 4) -> float:
        """
        Calculate velocity (change per week).
        
        Args:
            values: Series of values
            window: Window for velocity calculation
            
        Returns:
            Velocity value (change per week)
        """
        if len(values) < window:
            return 0.0
        
        recent_values = values.tail(window)
        if len(recent_values) < 2:
            return 0.0
        
        # Calculate change per week
        change = recent_values.iloc[-1] - recent_values.iloc[0]
        velocity = change / (len(recent_values) - 1)
        
        return velocity
    
    def _identify_risk_alerts(self, data: pd.DataFrame, brand_index: float, trend: TrendDirection) -> List[str]:
        """
        Identify risk alerts based on brand health indicators.
        
        Args:
            data: DataFrame with recent data
            brand_index: Current brand equity index
            trend: Current trend direction
            
        Returns:
            List of risk alert messages
        """
        alerts = []
        
        # Low brand index alert
        if brand_index < 30:
            alerts.append("Critical: Brand equity index below 30 - severe brand health concerns")
        elif brand_index < 50:
            alerts.append("Warning: Brand equity index below 50 - brand health deteriorating")
        
        # Negative trend alert
        if trend == TrendDirection.DECLINING:
            alerts.append("Alert: Brand equity declining for multiple weeks")
        
        # SLA consistency issues
        sla_variance = data['sla_days'].var()
        if sla_variance > 1.0:
            alerts.append("Warning: High SLA variance detected - customer trust at risk")
        
        # High discount intensity
        avg_discount = data['discount_pct'].mean()
        if avg_discount > 0.15:
            alerts.append("Warning: High discount intensity may erode brand equity")
        
        # Stockout issues
        avg_stockout = data['stockout_rate'].mean()
        if avg_stockout > 0.10:
            alerts.append("Warning: Frequent stockouts damaging brand reliability")
        
        return alerts
    
    def calculate_brand_equity(self, 
                               data: pd.DataFrame, 
                               mmm_model, 
                               time_range: TimeRange) -> BrandEquityMetrics:
        """
        Calculate Brand Equity Index and related metrics.
        
        Args:
            data: DataFrame with sales data
            mmm_model: Fitted MMM model
            time_range: Analysis time range
            
        Returns:
            BrandEquityMetrics with computed metrics
        """
        if len(data) < 8:
            raise ValueError("Need at least 8 weeks of data for brand equity calculation")
        
        # Calculate residuals from MMM
        residuals = self._calculate_residuals(data, mmm_model)
        
        # Apply smoothing
        smoothed_residuals = self._smooth_residuals(residuals)
        
        # Normalize to 0-100 index scale
        brand_index_series = self._normalize_to_index(smoothed_residuals)
        
        # Get current and previous values
        current_index = brand_index_series.iloc[-1]
        previous_index = brand_index_series.iloc[-2] if len(brand_index_series) > 1 else current_index
        
        # Calculate trend direction
        trend = self._calculate_trend_direction(brand_index_series)
        
        # Calculate velocity
        velocity = self._calculate_velocity(brand_index_series)
        
        # Calculate confidence level based on data quality
        data_quality_score = min(1.0, len(data) / 16)  # More data = higher confidence
        residual_stability = 1.0 - (residuals.std() / (residuals.mean() + 1e-6))
        confidence_level = (data_quality_score * 0.6 + residual_stability * 0.4)
        confidence_level = max(0.3, min(1.0, confidence_level))
        
        # Identify risk alerts
        recent_data = data.tail(4)  # Last 4 weeks
        risk_alerts = self._identify_risk_alerts(recent_data, current_index, trend)
        
        return BrandEquityMetrics(
            scope=data['sku_id'].unique().tolist(),
            time_range=time_range,
            brand_equity_index=current_index,
            previous_index=previous_index,
            trend_direction=trend,
            velocity=velocity,
            confidence_level=confidence_level,
            risk_alerts=risk_alerts
        )
    
    def analyze_brand_sensitivity(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze brand sensitivity to different factors.
        
        Args:
            data: DataFrame with operational data
            
        Returns:
            Dictionary mapping factors to sensitivity scores
        """
        sensitivity_scores = {}
        
        # SLA consistency sensitivity
        sla_variance = data['sla_days'].var()
        sensitivity_scores['sla_consistency'] = min(1.0, sla_variance * 0.5)
        
        # Procurement reliability sensitivity
        stockout_rate = data['stockout_rate'].mean()
        sensitivity_scores['procurement_reliability'] = min(1.0, stockout_rate * 5)
        
        # Price volatility sensitivity
        price_volatility = data['discount_pct'].std()
        sensitivity_scores['price_volatility'] = min(1.0, price_volatility * 3)
        
        # Discount intensity sensitivity
        avg_discount = data['discount_pct'].mean()
        sensitivity_scores['discount_intensity'] = min(1.0, avg_discount * 4)
        
        return sensitivity_scores
    
    def predict_brand_impact(self, 
                           interventions: Dict[str, float], 
                           current_sensitivity: Dict[str, float]) -> float:
        """
        Predict brand impact of interventions.
        
        Args:
            interventions: Dictionary of interventions and their magnitudes
            current_sensitivity: Current sensitivity scores
            
        Returns:
            Predicted brand equity impact (positive = improvement, negative = decline)
        """
        total_impact = 0.0
        
        for factor, change in interventions.items():
            if factor in current_sensitivity and factor in self.brand_sensitivity:
                # Calculate impact based on sensitivity and change
                sensitivity = current_sensitivity[factor]
                brand_sensitivity = self.brand_sensitivity[factor]
                
                # Impact = change * sensitivity * brand_sensitivity
                impact = change * sensitivity * brand_sensitivity
                total_impact += impact
        
        return total_impact
    
    def get_brand_health_recommendations(self, 
                                        brand_metrics: BrandEquityMetrics, 
                                        sensitivity_scores: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on brand health analysis.
        
        Args:
            brand_metrics: Computed brand equity metrics
            sensitivity_scores: Sensitivity analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Based on trend direction
        if brand_metrics.trend_direction == TrendDirection.DECLINING:
            recommendations.append("URGENT: Address declining brand equity trend")
        
        # Based on brand index level
        if brand_metrics.brand_equity_index < 50:
            recommendations.append("Focus on brand-building initiatives to restore health")
        
        # Based on sensitivity analysis
        high_sensitivity_factors = [factor for factor, score in sensitivity_scores.items() if score > 0.7]
        
        if 'sla_consistency' in high_sensitivity_factors:
            recommendations.append("Improve SLA consistency to build customer trust")
        
        if 'procurement_reliability' in high_sensitivity_factors:
            recommendations.append("Reduce stockouts to improve brand reliability")
        
        if 'discount_intensity' in high_sensitivity_factors:
            recommendations.append("Reduce discount dependency to protect brand equity")
        
        if 'price_volatility' in high_sensitivity_factors:
            recommendations.append("Stabilize pricing to maintain brand perception")
        
        # Based on risk alerts
        if "Critical: Brand equity index below 30" in brand_metrics.risk_alerts:
            recommendations.append("CRITICAL: Immediate brand recovery plan required")
        
        return recommendations
