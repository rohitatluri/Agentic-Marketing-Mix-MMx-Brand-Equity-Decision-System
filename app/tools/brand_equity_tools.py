"""Brand equity tools."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union
import logging

from ..data.synthetic import get_demo_data
from ..models.brand_equity import BrandEquityModel
from ..models.mmm import MarketingMixModel
from ..tools.schemas import BrandEquityMetrics, TimeRange

logger = logging.getLogger(__name__)


class BrandEquityTools:
    """Tools for brand equity analysis and monitoring."""
    
    def __init__(self):
        """Initialize brand equity tools."""
        self.demo_data = get_demo_data()
        self.brand_model = BrandEquityModel()
        self.mmm_tools = None  # Will be set to avoid circular import
        
    def set_mmm_tools(self, mmm_tools):
        """Set MMM tools reference (to avoid circular import)."""
        self.mmm_tools = mmm_tools
        
    def get_brand_equity(self, scope: List[str], time_range: TimeRange) -> BrandEquityMetrics:
        """
        Retrieve brand health metrics for specified SKUs and time range.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            BrandEquityMetrics with brand health indicators
        """
        # Validate inputs
        self._validate_inputs(scope, time_range)
        
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
            raise ValueError(f"Insufficient data for brand equity analysis. Need at least 3 weeks, got {len(combined_data)}")
        
        # If we have less than 6 weeks, use a simpler approach
        if len(combined_data) < 6:
            logger.warning(f"Using simplified brand equity analysis with {len(combined_data)} weeks (minimum 6 recommended)")
        
        # Get MMM model for residual calculation
        if self.mmm_tools is None:
            # Create a simple MMM model for residual calculation
            mmm_model = MarketingMixModel()
            mmm_model.fit(combined_data)
        else:
            # Get MMM model from tools
            model_key = tuple(sorted(scope))
            if model_key not in self.mmm_tools.mmm_models:
                self.mmm_tools.get_mmm_decomposition(scope, time_range)
            mmm_model = self.mmm_tools.mmm_models[model_key]
        
        # Calculate brand equity metrics
        try:
            brand_metrics = self.brand_model.calculate_brand_equity(combined_data, mmm_model, time_range)
        except Exception as e:
            logger.warning(f"Failed to calculate brand equity with MMM model: {e}. Using simplified approach.")
            # Create a simple brand equity metric based on available data
            brand_metrics = self._create_simple_brand_equity(combined_data, time_range)
        
        return brand_metrics
    
    def _create_simple_brand_equity(self, data: pd.DataFrame, time_range: TimeRange):
        """Create a simple brand equity metric when full calculation fails."""
        from ..tools.schemas import BrandEquityMetrics, TrendDirection
        
        logger.info("Creating simple brand equity based on data trends")
        
        # Use discount dependency as a proxy for brand health
        avg_discount = data['discount_pct'].mean()
        discount_trend = data['discount_pct'].iloc[-1] - data['discount_pct'].iloc[0]
        
        # Simple brand equity calculation (inverse of discount dependency)
        base_index = 70.0
        discount_penalty = avg_discount * 2  # 2 points per 1% discount
        trend_penalty = max(0, discount_trend) * 3  # Penalty for increasing discounts
        
        brand_index = max(20, min(90, base_index - discount_penalty - trend_penalty))
        
        # Determine trend
        if discount_trend > 1:
            trend = TrendDirection.DECLINING
        elif discount_trend < -1:
            trend = TrendDirection.IMPROVING
        else:
            trend = TrendDirection.STABLE
        
        # Simple velocity based on recent changes
        velocity = -discount_trend / 4  # Negative discount trend = positive brand velocity
        
        # Risk alerts
        alerts = []
        if avg_discount > 15:
            alerts.append("High discount dependency detected - brand at risk")
        if discount_trend > 2:
            alerts.append("Increasing discount trend - brand equity declining")
        
        return BrandEquityMetrics(
            scope=data['sku_id'].unique().tolist() if 'sku_id' in data.columns else [],
            time_range=time_range,
            brand_equity_index=brand_index,
            previous_index=brand_index - velocity,
            trend_direction=trend,
            velocity=velocity,
            confidence_level=0.5,  # Lower confidence for simple calculation
            risk_alerts=alerts
        )
    
    def _validate_inputs(self, scope: List[str], time_range: TimeRange):
        """Validate tool inputs."""
        if not scope:
            raise ValueError("Scope cannot be empty")
        
        if len(scope) > 100:
            raise ValueError("Scope cannot exceed 100 SKUs")
        
        if time_range.end_date <= time_range.start_date:
            raise ValueError("End date must be after start date")
        
        # Check if time range provides enough data points
        days_diff = (time_range.end_date - time_range.start_date).days
        weeks_diff = days_diff / 7
        if weeks_diff < 2:
            logger.warning(f"Time range is very short ({weeks_diff:.1f} weeks). Brand equity results may be less reliable.")
        
        # Note: We don't enforce minimum weeks here anymore, as we handle it in the main function
    
    def get_brand_sensitivity_analysis(self, scope: List[str], time_range: TimeRange) -> Dict[str, float]:
        """
        Analyze brand sensitivity to different factors.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            Dictionary mapping factors to sensitivity scores
        """
        # Get data
        all_data = []
        for sku_id in scope:
            if sku_id in self.demo_data:
                sku_data = self.demo_data[sku_id].copy()
                sku_data['week_date'] = pd.to_datetime(sku_data['week_date'])
                mask = (sku_data['week_date'] >= time_range.start_date) & \
                       (sku_data['week_date'] <= time_range.end_date)
                filtered_data = sku_data[mask]
                all_data.append(filtered_data)
        
        if not all_data:
            raise ValueError(f"No data found for SKUs: {scope}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Analyze sensitivity
        sensitivity_scores = self.brand_model.analyze_brand_sensitivity(combined_data)
        
        return sensitivity_scores
    
    def predict_brand_impact_of_interventions(self, 
                                             scope: List[str], 
                                             time_range: TimeRange,
                                             interventions: Dict[str, float]) -> float:
        """
        Predict brand impact of proposed interventions.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            interventions: Dictionary of interventions and their magnitudes
            
        Returns:
            Predicted brand equity impact
        """
        # Get current sensitivity
        sensitivity_scores = self.get_brand_sensitivity_analysis(scope, time_range)
        
        # Predict impact
        brand_impact = self.brand_model.predict_brand_impact(interventions, sensitivity_scores)
        
        return brand_impact
    
    def get_brand_health_recommendations(self, scope: List[str], time_range: TimeRange) -> List[str]:
        """
        Get brand health improvement recommendations.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            List of recommendations
        """
        # Get brand metrics
        brand_metrics = self.get_brand_equity(scope, time_range)
        
        # Get sensitivity analysis
        sensitivity_scores = self.get_brand_sensitivity_analysis(scope, time_range)
        
        # Generate recommendations
        recommendations = self.brand_model.get_brand_health_recommendations(brand_metrics, sensitivity_scores)
        
        return recommendations
    
    def get_brand_equity_trend(self, scope: List[str], time_range: TimeRange) -> pd.DataFrame:
        """
        Get brand equity trend over time.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            DataFrame with weekly brand equity values
        """
        # Get data
        all_data = []
        for sku_id in scope:
            if sku_id in self.demo_data:
                sku_data = self.demo_data[sku_id].copy()
                sku_data['week_date'] = pd.to_datetime(sku_data['week_date'])
                mask = (sku_data['week_date'] >= time_range.start_date) & \
                       (sku_data['week_date'] <= time_range.end_date)
                filtered_data = sku_data[mask]
                all_data.append(filtered_data)
        
        if not all_data:
            return pd.DataFrame()
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Get MMM model
        if self.mmm_tools is None:
            mmm_model = MarketingMixModel()
            mmm_model.fit(combined_data)
        else:
            model_key = tuple(sorted(scope))
            if model_key not in self.mmm_tools.mmm_models:
                self.mmm_tools.get_mmm_decomposition(scope, time_range)
            mmm_model = self.mmm_tools.mmm_models[model_key]
        
        # Calculate residuals and brand equity for each week
        residuals = self.brand_model._calculate_residuals(combined_data, mmm_model)
        smoothed_residuals = self.brand_model._smooth_residuals(residuals)
        brand_index_series = self.brand_model._normalize_to_index(smoothed_residuals)
        
        # Create trend DataFrame
        trend_data = pd.DataFrame({
            'week_number': combined_data['week_number'],
            'week_date': combined_data['week_date'],
            'brand_equity_index': brand_index_series
        })
        
        return trend_data
