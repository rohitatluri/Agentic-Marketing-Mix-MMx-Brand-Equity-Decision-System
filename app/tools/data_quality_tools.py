"""Data quality assessment tools."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union
import logging

from ..data.synthetic import get_demo_data
from ..tools.schemas import DataQualityReport, DataQualityIssue, TimeRange

logger = logging.getLogger(__name__)


class DataQualityTools:
    """Tools for data quality assessment and validation."""
    
    def __init__(self):
        """Initialize data quality tools."""
        self.demo_data = get_demo_data()
        
    def detect_data_issues(self, scope: List[str], time_range: TimeRange) -> DataQualityReport:
        """
        Pre-flight data quality check for specified SKUs and time range.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            DataQualityReport with quality assessment
        """
        # Validate inputs
        self._validate_inputs(scope, time_range)
        
        # Collect data for all SKUs in scope
        all_data = []
        missing_weeks = []
        
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
                
                # Check for missing weeks
                expected_weeks = pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq='W'
                )
                actual_weeks = set(filtered_data['week_date'].dt.date)
                missing_sku_weeks = [
                    week.date() for week in expected_weeks 
                    if week.date() not in actual_weeks
                ]
                missing_weeks.extend([f"{sku_id}: {week}" for week in missing_sku_weeks])
            else:
                missing_weeks.append(f"{sku_id}: No data available")
        
        if not all_data:
            return DataQualityReport(
                scope=scope,
                time_range=time_range,
                completeness_score=0.0,
                missing_weeks=missing_weeks,
                anomalies=[DataQualityIssue(
                    issue_type="No Data",
                    description=f"No data found for any SKU in scope: {scope}",
                    severity="high"
                )],
                supply_suppression_flags=[],
                recommendation="insufficient_data"
            )
        
        # Combine all SKU data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(combined_data, time_range, scope)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(combined_data)
        
        # Check for supply suppression
        supply_suppression_flags = self._detect_supply_suppression(combined_data)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(completeness_score, anomalies, supply_suppression_flags)
        
        return DataQualityReport(
            scope=scope,
            time_range=time_range,
            completeness_score=completeness_score,
            missing_weeks=missing_weeks,
            anomalies=anomalies,
            supply_suppression_flags=supply_suppression_flags,
            recommendation=recommendation
        )
    
    def _validate_inputs(self, scope: List[str], time_range: TimeRange):
        """Validate tool inputs."""
        if not scope:
            raise ValueError("Scope cannot be empty")
        
        if len(scope) > 100:
            raise ValueError("Scope cannot exceed 100 SKUs")
        
        if time_range.end_date <= time_range.start_date:
            raise ValueError("End date must be after start date")
    
    def _calculate_completeness_score(self, data: pd.DataFrame, time_range: TimeRange, scope: List[str]) -> float:
        """Calculate data completeness score."""
        # Expected data points
        expected_weeks = len(pd.date_range(
            start=time_range.start_date,
            end=time_range.end_date,
            freq='W'
        ))
        expected_data_points = expected_weeks * len(scope)
        
        # Actual data points
        actual_data_points = len(data)
        
        # Calculate completeness
        if expected_data_points == 0:
            return 0.0
        
        completeness = actual_data_points / expected_data_points
        
        # Penalize for missing values
        missing_values = data.isnull().sum().sum()
        total_values = len(data) * len(data.columns)
        if total_values > 0:
            completeness *= (1 - missing_values / total_values)
        
        return max(0.0, min(1.0, completeness))
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Detect data anomalies."""
        anomalies = []
        
        # Check for negative values where they shouldn't exist
        negative_gmv = (data['gmv'] < 0).sum()
        if negative_gmv > 0:
            anomalies.append(DataQualityIssue(
                issue_type="Negative GMV",
                description=f"Found {negative_gmv} records with negative GMV",
                severity="high"
            ))
        
        negative_units = (data['units'] < 0).sum()
        if negative_units > 0:
            anomalies.append(DataQualityIssue(
                issue_type="Negative Units",
                description=f"Found {negative_units} records with negative units",
                severity="high"
            ))
        
        # Check for extreme outliers
        for col in ['gmv', 'units', 'aov']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 3 * IQR)) | (data[col] > (Q3 + 3 * IQR))).sum()
                if outliers > 0:
                    severity = "high" if outliers > len(data) * 0.05 else "medium"
                    anomalies.append(DataQualityIssue(
                        issue_type=f"Extreme {col.upper()} Outliers",
                        description=f"Found {outliers} extreme outliers in {col}",
                        severity=severity
                    ))
        
        # Check for zero sales weeks
        zero_sales_weeks = (data['units'] == 0).sum()
        if zero_sales_weeks > len(data) * 0.2:  # More than 20% zero sales
            anomalies.append(DataQualityIssue(
                issue_type="Excessive Zero Sales",
                description=f"Found {zero_sales_weeks} zero sales weeks ({zero_sales_weeks/len(data)*100:.1f}%)",
                severity="medium"
            ))
        
        # Check for extreme discount values
        extreme_discounts = (data['discount_pct'] > 0.5).sum()
        if extreme_discounts > 0:
            anomalies.append(DataQualityIssue(
                issue_type="Extreme Discounts",
                description=f"Found {extreme_discounts} records with discounts > 50%",
                severity="medium"
            ))
        
        # Check for SLA issues
        high_sla = (data['sla_days'] > 14).sum()  # More than 2 weeks
        if high_sla > 0:
            anomalies.append(DataQualityIssue(
                issue_type="Extreme SLA",
                description=f"Found {high_sla} records with SLA > 14 days",
                severity="medium"
            ))
        
        # Check for data consistency
        if 'current_price' in data.columns and 'base_price' in data.columns:
            price_inconsistency = (data['current_price'] > data['base_price'] * 1.5).sum()
            if price_inconsistency > 0:
                anomalies.append(DataQualityIssue(
                    issue_type="Price Inconsistency",
                    description=f"Found {price_inconsistency} records where current price is 50%+ higher than base price",
                    severity="medium"
                ))
        
        return anomalies
    
    def _detect_supply_suppression(self, data: pd.DataFrame) -> List[str]:
        """Detect supply suppression indicators."""
        flags = []
        
        # High stockout rates
        high_stockout_skus = data.groupby('sku_id')['stockout_rate'].mean()
        high_stockout_skus = high_stockout_skus[high_stockout_skus > 0.3].index.tolist()
        if high_stockout_skus:
            flags.extend([f"High stockout rate: {sku}" for sku in high_stockout_skus])
        
        # Low marketing spend with high demand (potential supply constraint)
        low_marketing_high_demand = data[
            (data['marketing_spend'] < data['marketing_spend'].quantile(0.25)) &
            (data['units'] > data['units'].quantile(0.75))
        ]
        if len(low_marketing_high_demand) > len(data) * 0.1:
            flags.append("Potential supply constraint: Low marketing spend with high demand")
        
        # SLA degradation with stable demand
        sla_degradation = data.groupby('sku_id').apply(
            lambda x: (x['sla_days'].iloc[-1] - x['sla_days'].iloc[0]) > 2
        )
        degrading_skus = sla_degradation[sla_degradation].index.tolist()
        if degrading_skus:
            flags.extend([f"SLA degradation: {sku}" for sku in degrading_skus])
        
        return flags
    
    def _generate_recommendation(self, completeness_score: float, anomalies: List[DataQualityIssue], supply_flags: List[str]) -> str:
        """Generate data quality recommendation."""
        # Check for critical issues
        critical_anomalies = [a for a in anomalies if a.severity == "high"]
        
        if critical_anomalies:
            return "insufficient_data"
        
        if completeness_score < 0.4:
            return "insufficient_data"
        elif completeness_score < 0.6:
            return "caution"
        elif len(anomalies) > 5 or len(supply_flags) > 3:
            return "caution"
        else:
            return "proceed"
    
    def get_data_quality_summary(self, scope: List[str], time_range: TimeRange) -> Dict[str, Union[str, float, int]]:
        """
        Get a summary of data quality metrics.
        
        Args:
            scope: List of SKU IDs
            time_range: Analysis time range
            
        Returns:
            Dictionary with quality summary metrics
        """
        report = self.detect_data_issues(scope, time_range)
        
        return {
            "completeness_score": report.completeness_score,
            "total_anomalies": len(report.anomalies),
            "critical_anomalies": len([a for a in report.anomalies if a.severity == "high"]),
            "medium_anomalies": len([a for a in report.anomalies if a.severity == "medium"]),
            "missing_weeks_count": len(report.missing_weeks),
            "supply_flags_count": len(report.supply_suppression_flags),
            "recommendation": report.recommendation,
            "data_points": sum(len(self.demo_data.get(sku_id, pd.DataFrame())) for sku_id in scope)
        }
