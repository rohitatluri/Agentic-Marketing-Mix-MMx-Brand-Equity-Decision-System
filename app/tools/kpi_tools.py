"""KPI fetching tools."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union
import logging

from ..data.synthetic import get_demo_data
from ..tools.schemas import KPISummary, TimeRange

logger = logging.getLogger(__name__)


class KPITools:
    """Tools for fetching and analyzing KPI data."""
    
    def __init__(self):
        """Initialize KPI tools."""
        self.demo_data = get_demo_data()
    
    def get_kpi_summary(self, scope: List[str], time_range: TimeRange) -> KPISummary:
        """
        Fetch high-level KPI trends for specified SKUs and time range.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            KPISummary with KPI metrics and trends
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
        
        # Split into current and previous periods
        # For demo, we'll split the time range in half
        total_weeks = len(combined_data['week_number'].unique())
        split_week = total_weeks // 2
        
        previous_data = combined_data[combined_data['week_number'] <= split_week]
        current_data = combined_data[combined_data['week_number'] > split_week]
        
        # Calculate KPIs
        gmv_current = current_data['gmv'].sum()
        gmv_previous = previous_data['gmv'].sum()
        gmv_change_pct = ((gmv_current - gmv_previous) / gmv_previous * 100) if gmv_previous > 0 else 0
        
        units_current = int(current_data['units'].sum())
        units_previous = int(previous_data['units'].sum())
        units_change_pct = ((units_current - units_previous) / units_previous * 100) if units_previous > 0 else 0
        
        aov_current = gmv_current / units_current if units_current > 0 else 0
        aov_previous = gmv_previous / units_previous if units_previous > 0 else 0
        aov_change_pct = ((aov_current - aov_previous) / aov_previous * 100) if aov_previous > 0 else 0
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(combined_data)
        
        # Detect anomalies
        anomaly_flags = self._detect_anomalies(combined_data)
        
        return KPISummary(
            scope=scope,
            time_range=time_range,
            gmv_current=gmv_current,
            gmv_previous=gmv_previous,
            gmv_change_pct=gmv_change_pct,
            units_current=units_current,
            units_previous=units_previous,
            units_change_pct=units_change_pct,
            aov_current=aov_current,
            aov_previous=aov_previous,
            aov_change_pct=aov_change_pct,
            data_quality_score=data_quality_score,
            anomaly_flags=anomaly_flags
        )
    
    def _validate_inputs(self, scope: List[str], time_range: TimeRange):
        """Validate tool inputs."""
        if not scope:
            raise ValueError("Scope cannot be empty")
        
        if len(scope) > 100:
            raise ValueError("Scope cannot exceed 100 SKUs")
        
        if time_range.end_date <= time_range.start_date:
            raise ValueError("End date must be after start date")
        
        # Check if time range is reasonable (not too long)
        days_diff = (time_range.end_date - time_range.start_date).days
        if days_diff > 365:
            raise ValueError("Time range cannot exceed 1 year")
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness and consistency."""
        score = 1.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3
        
        # Check for duplicate weeks
        duplicate_weeks = data.duplicated(subset=['sku_id', 'week_number']).sum()
        if duplicate_weeks > 0:
            score -= min(0.2, duplicate_weeks / len(data))
        
        # Check for negative values where they shouldn't exist
        negative_gmv = (data['gmv'] < 0).sum()
        if negative_gmv > 0:
            score -= min(0.2, negative_gmv / len(data))
        
        negative_units = (data['units'] < 0).sum()
        if negative_units > 0:
            score -= min(0.2, negative_units / len(data))
        
        # Check for extreme outliers
        for col in ['gmv', 'units', 'aov']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 3 * IQR)) | (data[col] > (Q3 + 3 * IQR))).sum()
                if outliers > 0:
                    score -= min(0.1, outliers / len(data))
        
        return max(0.0, min(1.0, score))
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[str]:
        """Detect anomalies in the data."""
        anomalies = []
        
        # Check for large GMV swings
        if len(data) > 1:
            gmv_changes = data.groupby('sku_id')['gmv'].pct_change().abs()
            large_swings = gmv_changes[gmv_changes > 0.5].count()
            if large_swings > 0:
                anomalies.append(f"Large GMV swings detected in {large_swings} instances")
        
        # Check for zero sales weeks
        zero_sales_weeks = (data['units'] == 0).sum()
        if zero_sales_weeks > 0:
            anomalies.append(f"Zero sales weeks detected: {zero_sales_weeks}")
        
        # Check for extreme discount values
        extreme_discounts = (data['discount_pct'] > 0.3).sum()
        if extreme_discounts > 0:
            anomalies.append(f"Extreme discounts (>30%) detected: {extreme_discounts}")
        
        # Check for SLA degradation
        high_sla = (data['sla_days'] > 7).sum()
        if high_sla > 0:
            anomalies.append(f"High SLA (>7 days) detected: {high_sla}")
        
        # Check for stockout issues
        high_stockouts = (data['stockout_rate'] > 0.2).sum()
        if high_stockouts > 0:
            anomalies.append(f"High stockout rate (>20%) detected: {high_stockouts}")
        
        return anomalies
    
    def get_weekly_kpi_trends(self, scope: List[str], time_range: TimeRange) -> Dict[str, pd.DataFrame]:
        """
        Get weekly KPI trends for visualization.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            Dictionary with weekly trends data
        """
        weekly_data = []
        
        for sku_id in scope:
            if sku_id in self.demo_data:
                sku_data = self.demo_data[sku_id].copy()
                sku_data['week_date'] = pd.to_datetime(sku_data['week_date'])
                
                # Filter by time range - convert both to datetime for comparison
                start_date = pd.to_datetime(time_range.start_date)
                end_date = pd.to_datetime(time_range.end_date)
                mask = (sku_data['week_date'] >= start_date) & \
                       (sku_data['week_date'] <= end_date)
                filtered_data = sku_data[mask].copy()
                
                if not filtered_data.empty:
                    weekly_data.append(filtered_data)
        
        if not weekly_data:
            return {}
        
        # Combine data and aggregate by week
        combined_data = pd.concat(weekly_data, ignore_index=True)
        
        # Group by week across all SKUs
        weekly_trends = combined_data.groupby('week_number').agg({
            'gmv': 'sum',
            'units': 'sum',
            'aov': 'mean',
            'discount_pct': 'mean',
            'sla_days': 'mean',
            'stockout_rate': 'mean',
            'marketing_spend': 'sum',
            'week_date': 'first'
        }).reset_index()
        
        return {
            'weekly_trends': weekly_trends,
            'sku_breakdown': combined_data.groupby(['week_number', 'sku_id']).agg({
                'gmv': 'sum',
                'units': 'sum'
            }).reset_index()
        }
    
    def compare_sku_performance(self, scope: List[str], time_range: TimeRange) -> pd.DataFrame:
        """
        Compare performance across SKUs.
        
        Args:
            scope: List of SKU IDs to compare
            time_range: Analysis time range
            
        Returns:
            DataFrame with SKU comparison metrics
        """
        sku_metrics = []
        
        for sku_id in scope:
            if sku_id in self.demo_data:
                sku_data = self.demo_data[sku_id].copy()
                sku_data['week_date'] = pd.to_datetime(sku_data['week_date'])
                
                # Filter by time range - convert both to datetime for comparison
                start_date = pd.to_datetime(time_range.start_date)
                end_date = pd.to_datetime(time_range.end_date)
                mask = (sku_data['week_date'] >= start_date) & \
                       (sku_data['week_date'] <= end_date)
                filtered_data = sku_data[mask]
                
                if not filtered_data.empty:
                    # Calculate metrics for this SKU
                    total_gmv = filtered_data['gmv'].sum()
                    total_units = filtered_data['units'].sum()
                    avg_aov = filtered_data['aov'].mean()
                    avg_discount = filtered_data['discount_pct'].mean()
                    avg_sla = filtered_data['sla_days'].mean()
                    avg_stockout = filtered_data['stockout_rate'].mean()
                    
                    sku_metrics.append({
                        'sku_id': sku_id,
                        'total_gmv': total_gmv,
                        'total_units': total_units,
                        'avg_aov': avg_aov,
                        'avg_discount': avg_discount,
                        'avg_sla_days': avg_sla,
                        'avg_stockout_rate': avg_stockout,
                        'weeks_of_data': len(filtered_data)
                    })
        
        return pd.DataFrame(sku_metrics)
