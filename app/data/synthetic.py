"""Demo data generator for testing and development."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple
import random


class SyntheticDataGenerator:
    """Generates realistic synthetic data for MMM agent testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        
        # SKU definitions with realistic characteristics
        self.skus = {
            "SKU-123": {
                "name": "Premium Coffee Maker",
                "base_price": 150.0,
                "base_weekly_demand": 280,
                "price_elasticity": -1.8,
                "sla_sensitivity": -0.4,
                "procurement_sensitivity": -0.3,
                "marketing_elasticity": 0.6
            },
            "SKU-456": {
                "name": "Wireless Headphones",
                "base_price": 89.99,
                "base_weekly_demand": 450,
                "price_elasticity": -2.2,
                "sla_sensitivity": -0.5,
                "procurement_sensitivity": -0.4,
                "marketing_elasticity": 0.8
            },
            "SKU-789": {
                "name": "Smart Watch",
                "base_price": 299.99,
                "base_weekly_demand": 180,
                "price_elasticity": -1.5,
                "sla_sensitivity": -0.3,
                "procurement_sensitivity": -0.2,
                "marketing_elasticity": 0.7
            }
        }
    
    def generate_weekly_data(self, 
                           sku_id: str, 
                           start_date: date, 
                           weeks: int = 16,
                           scenario: str = "normal") -> pd.DataFrame:
        """
        Generate weekly data for a specific SKU.
        
        Args:
            sku_id: SKU identifier
            start_date: Start date for data generation
            weeks: Number of weeks to generate
            scenario: Scenario type ('normal', 'sla_degradation', 'procurement_issues', 'marketing_campaign')
        
        Returns:
            DataFrame with weekly data
        """
        if sku_id not in self.skus:
            raise ValueError(f"Unknown SKU: {sku_id}")
        
        sku_config = self.skus[sku_id]
        dates = [start_date + timedelta(weeks=i) for i in range(weeks)]
        
        data = []
        for i, week_date in enumerate(dates):
            # Base demand with trend and seasonality
            trend = 1.0 + (i * 0.002)  # Slight upward trend
            seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * i / 52)  # Yearly seasonality
            base_demand = sku_config["base_weekly_demand"] * trend * seasonal
            
            # Generate driver values based on scenario
            if scenario == "sla_degradation" and i >= 8:
                # SLA degrades after week 8
                sla_days = 3 + (i - 7) * 0.3  # Gradually worsen
                sla_success_rate = max(0.7, 0.95 - (i - 7) * 0.02)
            else:
                sla_days = np.random.normal(3, 0.5)
                sla_success_rate = np.clip(np.random.normal(0.95, 0.02), 0.85, 0.99)
            
            if scenario == "procurement_issues" and i >= 10:
                # Procurement issues after week 10
                procurement_sla = 7 + (i - 9) * 0.5
                stockout_rate = min(0.25, 0.05 + (i - 9) * 0.02)
            else:
                procurement_sla = np.random.normal(5, 1)
                stockout_rate = np.clip(np.random.normal(0.05, 0.02), 0, 0.15)
            
            # Discount strategy
            if scenario == "marketing_campaign" and 6 <= i <= 10:
                discount = np.random.uniform(0.15, 0.25)  # Higher discount during campaign
            else:
                discount = np.clip(np.random.normal(0.08, 0.03), 0, 0.20)
            
            # Marketing spend
            if scenario == "marketing_campaign" and 6 <= i <= 10:
                marketing_spend = np.random.uniform(8000, 12000)  # Campaign period
            else:
                marketing_spend = np.random.uniform(2000, 4000)  # Normal period
            
            # Calculate price after discount
            current_price = sku_config["base_price"] * (1 - discount)
            
            # Calculate demand using elasticities
            price_impact = (current_price / sku_config["base_price"]) ** sku_config["price_elasticity"]
            sla_impact = (sla_days / 3) ** sku_config["sla_sensitivity"]
            procurement_impact = (1 - stockout_rate) ** sku_config["procurement_sensitivity"]
            marketing_impact = 1 + (marketing_spend / 3000) * sku_config["marketing_elasticity"] * 0.01
            
            # Add some random noise
            noise = np.random.normal(1.0, 0.05)
            
            # Final demand
            demand = base_demand * price_impact * sla_impact * procurement_impact * marketing_impact * noise
            demand = max(0, int(demand))
            
            # Calculate financial metrics
            gmv = demand * current_price
            units = demand
            aov = gmv / units if units > 0 else 0
            
            data.append({
                "week_date": week_date,
                "week_number": i + 1,
                "sku_id": sku_id,
                "gmv": gmv,
                "units": units,
                "aov": aov,
                "base_price": sku_config["base_price"],
                "current_price": current_price,
                "discount_pct": discount,
                "sla_days": sla_days,
                "sla_success_rate": sla_success_rate,
                "procurement_sla_days": procurement_sla,
                "stockout_rate": stockout_rate,
                "marketing_spend": marketing_spend,
                "competitor_price_index": np.random.normal(1.0, 0.05)
            })
        
        return pd.DataFrame(data)
    
    def generate_dataset(self, weeks: int = 16) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset for all SKUs."""
        start_date = date.today() - timedelta(weeks=weeks)
        
        datasets = {}
        
        # Generate different scenarios for different SKUs to showcase variety
        datasets["SKU-123"] = self.generate_weekly_data("SKU-123", start_date, weeks, "sla_degradation")
        datasets["SKU-456"] = self.generate_weekly_data("SKU-456", start_date, weeks, "procurement_issues")
        datasets["SKU-789"] = self.generate_weekly_data("SKU-789", start_date, weeks, "marketing_campaign")
        
        return datasets
    
    def get_sku_info(self, sku_id: str) -> Dict:
        """Get SKU configuration information."""
        return self.skus.get(sku_id, {})
    
    def create_sample_queries(self) -> List[str]:
        """Create sample user queries for testing."""
        return [
            "Why did GMV drop for SKU-123 in the last 4 weeks?",
            "What should we do next week to recover sales for SKU-123?",
            "What if we reduce discount by 5% but improve SLA by 1 day?",
            "Analyze SKU-456 GMV performance",
            "Should we increase discount to 25% to boost sales for SKU-789?",
            "Compare performance across all SKUs",
            "What's driving the brand equity decline for SKU-123?"
        ]


# Global instance for easy access
data_generator = SyntheticDataGenerator()

# Pre-generate demo data
demo_data = data_generator.generate_dataset(weeks=24)

def get_demo_data(sku_id: str = None) -> Dict[str, pd.DataFrame]:
    """Get demo data for specific SKU or all SKUs."""
    if sku_id:
        return {sku_id: demo_data.get(sku_id, pd.DataFrame())}
    return demo_data

def get_sample_queries() -> List[str]:
    """Get sample queries for testing."""
    return data_generator.create_sample_queries()
