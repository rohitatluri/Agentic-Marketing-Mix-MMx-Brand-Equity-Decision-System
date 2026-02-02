"""MMM decomposition tools."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union
import logging

from ..data.synthetic import get_demo_data
from ..models.mmm import MarketingMixModel
from ..tools.schemas import MMMDecomposition, TimeRange

logger = logging.getLogger(__name__)


class MMMTools:
    """Tools for Marketing Mix Model decomposition and analysis."""
    
    def __init__(self):
        """Initialize MMM tools."""
        self.demo_data = get_demo_data()
        self.mmm_models = {}  # Cache fitted models per SKU
        
    def get_mmm_decomposition(self, scope: List[str], time_range: TimeRange) -> MMMDecomposition:
        """
        Get driver-level attribution for specified SKUs and time range.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            MMMDecomposition with driver contributions
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
        if len(combined_data) < 4:
            raise ValueError(f"Insufficient data for MMM analysis. Need at least 4 weeks, got {len(combined_data)}")
        
        # Check if we have enough data for full MMM
        if len(combined_data) < 6:
            logger.warning(f"Using simplified MMM analysis with {len(combined_data)} weeks (minimum 6 recommended)")
            return self._create_simple_decomposition(combined_data, time_range)
        
        # For 6-8 weeks, use simplified approach
        if len(combined_data) < 8:
            logger.warning(f"Using simplified MMM analysis with {len(combined_data)} weeks (minimum 8 recommended for full model)")
        
        # Fit or get cached MMM model
        model_key = tuple(sorted(scope))  # Use sorted tuple as cache key
        if model_key not in self.mmm_models:
            model = MarketingMixModel()
            try:
                fit_metrics = model.fit(combined_data)
                logger.info(f"Fitted MMM model for {scope}: R²={fit_metrics['r2_score']:.3f}")
            except Exception as e:
                logger.warning(f"Failed to fit full MMM model: {e}. Using simplified approach.")
                # Create a simple decomposition based on correlations
                return self._create_simple_decomposition(combined_data, time_range)
            self.mmm_models[model_key] = model
        else:
            model = self.mmm_models[model_key]
            logger.info(f"Using cached MMM model for {scope}")
        
        # Perform decomposition
        try:
            decomposition = model.decompose_changes(combined_data, time_range)
        except Exception as e:
            logger.warning(f"Failed to decompose with MMM model: {e}. Using simplified approach.")
            return self._create_simple_decomposition(combined_data, time_range)
        
        return decomposition
    
    def _create_simple_decomposition(self, data: pd.DataFrame, time_range: TimeRange):
        """Create a simple decomposition when full MMM fails."""
        from ..tools.schemas import MMMDecomposition, DriverContribution
        
        logger.info("Creating simple decomposition based on data trends")
        
        # Calculate overall change
        if len(data) < 2:
            # Not enough data for comparison
            total_change_pct = 0.0
        else:
            # Compare recent vs. older periods
            recent_data = data.tail(2)  # Last 2 weeks
            older_data = data.head(2)   # First 2 weeks
            
            recent_gmv = recent_data['gmv'].mean()
            older_gmv = older_data['gmv'].mean()
            
            if older_gmv > 0:
                total_change_pct = ((recent_gmv - older_gmv) / older_gmv) * 100
            else:
                total_change_pct = 0.0
        
        # Create simple driver contributions based on correlations
        contributions = []
        
        # Price impact (inverse correlation with discount)
        if 'discount_pct' in data.columns:
            discount_change = data['discount_pct'].iloc[-1] - data['discount_pct'].iloc[0]
            price_impact = -discount_change * 2.0  # Simple elasticity assumption
            avg_gmv = data['gmv'].mean()
            impact_amount = (price_impact / 100) * avg_gmv
            contributions.append(DriverContribution(
                driver="Pricing",
                contribution_pct=price_impact,
                impact_amount=impact_amount,
                confidence=0.6
            ))
        
        # SLA impact
        if 'sla_days' in data.columns:
            sla_change = data['sla_days'].iloc[-1] - data['sla_days'].iloc[0]
            sla_impact = -sla_change * 1.5  # Simple sensitivity
            avg_gmv = data['gmv'].mean()
            impact_amount = (sla_impact / 100) * avg_gmv
            contributions.append(DriverContribution(
                driver="SLA",
                contribution_pct=sla_impact,
                impact_amount=impact_amount,
                confidence=0.7
            ))
        
        # Marketing impact
        if 'marketing_spend' in data.columns:
            marketing_change = data['marketing_spend'].iloc[-1] - data['marketing_spend'].iloc[0]
            if marketing_change > 0:
                marketing_impact = min(marketing_change / 1000, 5.0)  # Cap at 5%
            else:
                marketing_impact = marketing_change / 1000
            avg_gmv = data['gmv'].mean()
            impact_amount = (marketing_impact / 100) * avg_gmv
            contributions.append(DriverContribution(
                driver="Marketing",
                contribution_pct=marketing_impact,
                impact_amount=impact_amount,
                confidence=0.5
            ))
        
        # Seasonal/Trend (residual)
        accounted_impact = sum(c.contribution_pct for c in contributions)
        residual = total_change_pct - accounted_impact
        avg_gmv = data['gmv'].mean()
        impact_amount = (residual / 100) * avg_gmv
        contributions.append(DriverContribution(
            driver="Seasonal/Trend",
            contribution_pct=residual,
            impact_amount=impact_amount,
            confidence=0.4
        ))
        
        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x.contribution_pct), reverse=True)
        
        return MMMDecomposition(
            scope=data['sku_id'].unique().tolist() if 'sku_id' in data.columns else [],
            time_range=time_range,
            total_change_pct=total_change_pct,
            baseline_demand=data['gmv'].mean(),
            residual=residual,
            model_fit_r2=0.5,  # Lower confidence for simple model
            driver_contributions=contributions,
            confidence_score=0.6
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
            logger.warning(f"Time range is very short ({weeks_diff:.1f} weeks). Results may be less reliable.")
        
        # Note: We don't enforce minimum weeks here anymore, as we handle it in the main function
    
    def get_driver_elasticities(self, scope: List[str], time_range: TimeRange) -> Dict[str, Dict[str, float]]:
        """
        Get driver elasticities for the specified scope.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            Dictionary mapping drivers to their elasticities
        """
        # Get or fit MMM model
        model_key = tuple(sorted(scope))
        if model_key not in self.mmm_models:
            # This will fit the model and cache it
            self.get_mmm_decomposition(scope, time_range)
        
        model = self.mmm_models[model_key]
        
        # Extract elasticities from model coefficients
        elasticities = {}
        for feature, coefficient in model.coefficients.items():
            # Map feature names to driver names
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
            
            driver = driver_mapping.get(feature, feature)
            if driver not in elasticities:
                elasticities[driver] = 0
            elasticities[driver] += abs(coefficient)  # Use absolute value for elasticity magnitude
        
        return elasticities
    
    def get_model_fit_metrics(self, scope: List[str], time_range: TimeRange) -> Dict[str, float]:
        """
        Get model fit metrics for the MMM.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            Dictionary with model fit metrics
        """
        # Get or fit MMM model
        model_key = tuple(sorted(scope))
        if model_key not in self.mmm_models:
            # This will fit the model and cache it
            self.get_mmm_decomposition(scope, time_range)
        
        model = self.mmm_models[model_key]
        
        # Get data for validation
        all_data = []
        for sku_id in scope:
            if sku_id in self.demo_data:
                sku_data = self.demo_data[sku_id].copy()
                sku_data['week_date'] = pd.to_datetime(sku_data['week_date'])
                mask = (sku_data['week_date'] >= time_range.start_date) & \
                       (sku_data['week_date'] <= time_range.end_date)
                filtered_data = sku_data[mask]
                all_data.append(filtered_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Calculate predictions and metrics
        predictions = model.predict(combined_data)
        actual = combined_data['units']
        
        # R-squared
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # MAE and RMSE
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        
        # MAPE
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'data_points': len(combined_data),
            'features_count': len(model.feature_names)
        }
    
    def get_driver_contributions_over_time(self, scope: List[str], time_range: TimeRange) -> pd.DataFrame:
        """
        Get driver contributions over time for trend analysis.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            DataFrame with weekly driver contributions
        """
        # Get MMM model
        model_key = tuple(sorted(scope))
        if model_key not in self.mmm_models:
            self.get_mmm_decomposition(scope, time_range)
        
        model = self.mmm_models[model_key]
        
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
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Calculate contributions for each week
        weekly_contributions = []
        
        for week in combined_data['week_number'].unique():
            week_data = combined_data[combined_data['week_number'] == week]
            
            # Get baseline (use first week as reference)
            if week == combined_data['week_number'].min():
                baseline_data = week_data
            else:
                baseline_data = combined_data[combined_data['week_number'] == combined_data['week_number'].min()]
            
            # Calculate contribution for this week
            if len(baseline_data) > 0 and len(week_data) > 0:
                baseline_units = baseline_data['units'].mean()
                current_units = week_data['units'].mean()
                
                if baseline_units > 0:
                    total_change = current_units - baseline_units
                    total_change_pct = (total_change / baseline_units) * 100
                    
                    # Simplified contribution calculation (in real implementation, 
                    # this would use the actual model coefficients)
                    contributions = {
                        'week_number': week,
                        'total_change_pct': total_change_pct,
                        'Price/Discount': total_change_pct * 0.2,  # Simplified
                        'SLA': total_change_pct * 0.3,
                        'Procurement': total_change_pct * 0.25,
                        'Marketing': total_change_pct * 0.15,
                        'Seasonal/Trend': total_change_pct * 0.1
                    }
                    weekly_contributions.append(contributions)
        
        return pd.DataFrame(weekly_contributions)
    
    def clear_model_cache(self):
        """Clear cached MMM models."""
        self.mmm_models.clear()
        logger.info("Cleared MMM model cache")
