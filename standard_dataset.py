"""
Standard Comprehensive Dataset for MMM Agent System
Contains realistic business data with multiple scenarios and patterns
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List
import random

class StandardDatasetGenerator:
    """Generate a comprehensive standard dataset for testing."""
    
    def __init__(self):
        self.skus = {
            # Electronics Category
            "ELEC-001": {
                "name": "Premium Laptop",
                "category": "Electronics",
                "base_price": 1200.0,
                "base_weekly_demand": 150,
                "price_elasticity": -1.8,
                "sla_sensitivity": 0.7,
                "marketing_efficiency": 0.4
            },
            "ELEC-002": {
                "name": "Smartphone Pro",
                "category": "Electronics", 
                "base_price": 800.0,
                "base_weekly_demand": 300,
                "price_elasticity": -2.2,
                "sla_sensitivity": 0.8,
                "marketing_efficiency": 0.5
            },
            "ELEC-003": {
                "name": "Wireless Headphones",
                "category": "Electronics",
                "base_price": 150.0,
                "base_weekly_demand": 500,
                "price_elasticity": -1.5,
                "sla_sensitivity": 0.6,
                "marketing_efficiency": 0.3
            },
            
            # Fashion Category
            "FASH-001": {
                "name": "Designer Jacket",
                "category": "Fashion",
                "base_price": 250.0,
                "base_weekly_demand": 200,
                "price_elasticity": -2.0,
                "sla_sensitivity": 0.9,
                "marketing_efficiency": 0.6
            },
            "FASH-002": {
                "name": "Casual Sneakers",
                "category": "Fashion",
                "base_price": 120.0,
                "base_weekly_demand": 400,
                "price_elasticity": -1.7,
                "sla_sensitivity": 0.7,
                "marketing_efficiency": 0.4
            },
            "FASH-003": {
                "name": "Premium Watch",
                "category": "Fashion",
                "base_price": 450.0,
                "base_weekly_demand": 80,
                "price_elasticity": -1.3,
                "sla_sensitivity": 0.8,
                "marketing_efficiency": 0.5
            },
            
            # Home & Garden
            "HOME-001": {
                "name": "Coffee Maker Deluxe",
                "category": "Home",
                "base_price": 180.0,
                "base_weekly_demand": 250,
                "price_elasticity": -1.6,
                "sla_sensitivity": 0.5,
                "marketing_efficiency": 0.3
            },
            "HOME-002": {
                "name": "Smart Thermostat",
                "category": "Home",
                "base_price": 300.0,
                "base_weekly_demand": 120,
                "price_elasticity": -1.4,
                "sla_sensitivity": 0.6,
                "marketing_efficiency": 0.4
            },
            "HOME-003": {
                "name": "Garden Tool Set",
                "category": "Home",
                "base_price": 85.0,
                "base_weekly_demand": 350,
                "price_elasticity": -1.8,
                "sla_sensitivity": 0.4,
                "marketing_efficiency": 0.2
            },
            
            # Sports & Outdoors
            "SPORT-001": {
                "name": "Yoga Mat Premium",
                "category": "Sports",
                "base_price": 45.0,
                "base_weekly_demand": 600,
                "price_elasticity": -1.9,
                "sla_sensitivity": 0.3,
                "marketing_efficiency": 0.3
            },
            "SPORT-002": {
                "name": "Running Shoes Pro",
                "category": "Sports",
                "base_price": 140.0,
                "base_weekly_demand": 280,
                "price_elasticity": -1.7,
                "sla_sensitivity": 0.6,
                "marketing_efficiency": 0.5
            },
            "SPORT-003": {
                "name": "Fitness Tracker",
                "category": "Sports",
                "base_price": 95.0,
                "base_weekly_demand": 450,
                "price_elasticity": -1.6,
                "sla_sensitivity": 0.5,
                "marketing_efficiency": 0.4
            }
        }
        
        # Seasonal patterns (multipliers for different months)
        self.seasonal_patterns = {
            "Electronics": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.5, 1.8],  # Peak in Nov-Dec
            "Fashion": [0.7, 0.8, 1.1, 1.2, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2, 1.3, 1.6],  # Spring & Fall
            "Home": [0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2, 1.3],  # Spring & Holiday
            "Sports": [0.8, 0.9, 1.2, 1.4, 1.5, 1.6, 1.5, 1.3, 1.2, 1.1, 0.9, 0.8]  # Summer peak
        }
    
    def generate_comprehensive_dataset(self, weeks: int = 52) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive dataset for all SKUs."""
        start_date = date.today() - timedelta(weeks=weeks)
        
        datasets = {}
        
        # Generate different scenarios for different SKUs
        scenarios = {
            "ELEC-001": "mixed_performance",  # Complex pattern
            "ELEC-002": "growth_trend",       # Growing trend
            "ELEC-003": "seasonal_spike",     # Seasonal variations
            "FASH-001": "price_sensitive",    # Price sensitive
            "FASH-002": "sla_issues",         # SLA problems
            "FASH-003": "premium_decline",   # Premium segment decline
            "HOME-001": "marketing_driven",   # Marketing driven
            "HOME-002": "stable_growth",      # Stable growth
            "HOME-003": "discount_heavy",     # Discount dependent
            "SPORT-001": "volatile_demand",    # Volatile demand
            "SPORT-002": "competitive_pressure", # Competitive pressure
            "SPORT-003": "brand_building"     # Brand building phase
        }
        
        for sku_id, scenario in scenarios.items():
            datasets[sku_id] = self.generate_weekly_data(sku_id, start_date, weeks, scenario)
        
        return datasets
    
    def generate_weekly_data(self, sku_id: str, start_date: date, weeks: int, scenario: str) -> pd.DataFrame:
        """Generate weekly data for a specific SKU with realistic patterns."""
        sku_config = self.skus[sku_id]
        
        # Create date range
        dates = [start_date + timedelta(weeks=i) for i in range(weeks)]
        
        # Initialize arrays
        base_demand = sku_config['base_weekly_demand']
        base_price = sku_config['base_price']
        category = sku_config['category']
        
        demand = []
        price = []
        discount = []
        sla = []
        marketing_spend = []
        gmv = []
        stockout_rate = []
        
        for i, week_date in enumerate(dates):
            month = week_date.month - 1  # 0-indexed
            seasonal_factor = self.seasonal_patterns[category][month]
            
            # Base demand with seasonal adjustment
            week_demand = base_demand * seasonal_factor
            
            # Apply scenario-specific patterns
            if scenario == "mixed_performance":
                # Complex pattern with multiple factors
                trend_factor = 1.0 + (i * 0.002)  # Slight upward trend
                if i > 20 and i < 30:  # Mid-year dip
                    trend_factor *= 0.85
                elif i > 40:  # Year-end boost
                    trend_factor *= 1.2
                week_demand *= trend_factor
                
                # Price strategy
                week_price = base_price * (1.0 + np.sin(i * 0.3) * 0.1)
                week_discount = max(0, min(30, 10 + np.sin(i * 0.2) * 8))
                
                # SLA degradation in middle period
                week_sla = 2.0 if 15 < i < 35 else 1.5
                
                # Marketing campaigns
                week_marketing = 5000 + np.sin(i * 0.15) * 2000
                
            elif scenario == "growth_trend":
                # Steady growth with occasional promotions
                growth_factor = 1.0 + (i * 0.008)  # 0.8% weekly growth
                week_demand *= growth_factor
                
                week_price = base_price * (0.95 + i * 0.001)  # Slight price increase
                week_discount = 5 + (i % 10)  # Cyclical discounts
                
                week_sla = 1.2  # Good SLA
                week_marketing = 3000 + i * 50  # Increasing marketing
                
            elif scenario == "seasonal_spike":
                # Strong seasonal patterns
                week_demand *= seasonal_factor
                if month in [10, 11]:  # Holiday season
                    week_demand *= 1.5
                
                week_price = base_price * (1.1 if month in [10, 11] else 1.0)
                week_discount = 15 if month in [10, 11] else 8
                
                week_sla = 2.5 if month in [11, 12] else 1.8  # Holiday delays
                week_marketing = 8000 if month in [10, 11] else 2000
                
            elif scenario == "price_sensitive":
                # High price sensitivity
                price_elasticity = sku_config['price_elasticity']
                week_discount = 5 + (i % 15)  # Variable discounts
                price_impact = (week_discount / 100) * price_elasticity
                week_demand *= (1 + price_impact)
                
                week_price = base_price * (1 - week_discount / 100)
                week_sla = 1.5
                week_marketing = 1500
                
            elif scenario == "sla_issues":
                # SLA problems affecting demand
                week_sla = 1.0 + (i % 8) * 0.3  # Deteriorating SLA
                sla_impact = (week_sla - 1.0) * sku_config['sla_sensitivity'] * 0.1
                week_demand *= (1 - sla_impact)
                
                week_price = base_price * 1.05  # Slight price premium
                week_discount = 5
                week_marketing = 2000
                
            elif scenario == "premium_decline":
                # Premium segment facing challenges
                decline_factor = 1.0 - (i * 0.005)  # Gradual decline
                week_demand *= decline_factor
                
                week_price = base_price * (1.1 - i * 0.002)  # Price reduction attempts
                week_discount = 10 + i * 0.3
                
                week_sla = 1.0  # Premium service
                week_marketing = 4000 + (weeks - i) * 100  # Increased marketing
                
            elif scenario == "marketing_driven":
                # Marketing-driven demand
                marketing_cycles = np.sin(i * 0.25) * 0.5 + 0.5  # 0 to 1
                week_demand *= (1 + marketing_cycles * 0.4)
                
                week_price = base_price
                week_discount = 8
                
                week_sla = 1.8
                week_marketing = 1000 + marketing_cycles * 8000
                
            elif scenario == "stable_growth":
                # Consistent stable growth
                stable_growth = 1.0 + (i * 0.003)
                week_demand *= stable_growth
                
                week_price = base_price * (1 + i * 0.0005)
                week_discount = 7
                
                week_sla = 1.3
                week_marketing = 2500 + i * 20
                
            elif scenario == "discount_heavy":
                # Heavy discount dependency
                week_discount = 15 + (i % 20)  # High discounts
                discount_impact = (week_discount / 100) * abs(sku_config['price_elasticity'])
                week_demand *= (1 + discount_impact * 0.7)
                
                week_price = base_price * (1 - week_discount / 100)
                week_sla = 2.0  # Average SLA
                week_marketing = 1000
                
            elif scenario == "volatile_demand":
                # High volatility
                volatility = np.sin(i * 0.5) * 0.3 + np.random.normal(0, 0.1)
                week_demand *= (1 + volatility)
                
                week_price = base_price * (1 + np.random.normal(0, 0.05))
                week_discount = 10 + np.random.normal(0, 3)
                week_discount = max(0, min(25, week_discount))
                
                week_sla = 1.0 + np.random.normal(0, 0.5)
                week_sla = max(0.5, min(5.0, week_sla))
                week_marketing = 2000 + np.random.normal(0, 500)
                
            elif scenario == "competitive_pressure":
                # Facing competitive pressure
                pressure_factor = 1.0 - (i * 0.002)  # Gradual share loss
                week_demand *= pressure_factor
                
                week_price = base_price * (0.9 - i * 0.001)  # Price cuts
                week_discount = 12 + i * 0.2
                
                week_sla = 1.0  # Competitive SLA
                week_marketing = 3000 + (weeks - i) * 150
                
            elif scenario == "brand_building":
                # Brand building phase
                brand_investment = np.sin(i * 0.2) * 0.3 + 0.7
                week_demand *= (1 + brand_investment * 0.2)
                
                week_price = base_price * 1.1  # Premium positioning
                week_discount = 5  # Low discounts
                
                week_sla = 1.0  # Premium service
                week_marketing = 2000 + brand_investment * 6000
                
            else:
                # Default pattern
                week_demand *= (1 + np.sin(i * 0.1) * 0.1)
                week_price = base_price
                week_discount = 10
                week_sla = 1.5
                week_marketing = 3000
            
            # Add some random noise
            week_demand *= (1 + np.random.normal(0, 0.05))
            week_demand = max(10, week_demand)  # Minimum demand
            
            # Calculate stockout rate (random but realistic)
            week_stockout_rate = max(0, min(1, np.random.beta(2, 20) + (i * 0.001)))  # Slight increase over time
            
            # Calculate final values
            week_units = int(week_demand)
            week_gmv_value = week_units * week_price * (1 - week_discount / 100)
            
            # Store values
            demand.append(week_units)
            price.append(week_price)
            discount.append(week_discount)
            sla.append(week_sla)
            marketing_spend.append(max(0, week_marketing))
            gmv.append(week_gmv_value)
            stockout_rate.append(week_stockout_rate)
        
        # Create DataFrame
        data = pd.DataFrame({
            'week_date': dates,
            'sku_id': sku_id,
            'gmv': gmv,
            'units': demand,
            'price': price,
            'discount_pct': discount,
            'sla_days': sla,
            'marketing_spend': marketing_spend,
            'stockout_rate': stockout_rate,
            'category': category,
            'scenario': scenario
        })
        
        # Add week_number (1-52)
        data['week_number'] = range(1, len(data) + 1)
        
        return data
    
    def create_standard_queries(self) -> List[str]:
        """Create comprehensive test queries for the standard dataset."""
        return [
            # Performance analysis queries
            "Analyze overall performance across all categories",
            "Compare Electronics vs Fashion category performance",
            "Which SKUs are showing the strongest growth trend?",
            "What's driving the decline in premium segments?",
            
            # Root cause analysis
            "Why are discount-heavy SKUs underperforming?",
            "What's causing SLA issues in Fashion category?",
            "Analyze the impact of marketing spend on sales",
            "Why is there volatile demand in Sports category?",
            
            # Scenario analysis
            "What if we reduce discounts by 10% across all SKUs?",
            "Impact of improving SLA to 1 day for premium products",
            "Should we increase marketing spend by 20%?",
            "Effect of price increase on price-sensitive SKUs",
            
            # Strategic questions
            "Which categories need immediate attention?",
            "What are the top 3 recommendations for growth?",
            "How to reduce discount dependency?",
            "Brand equity analysis for premium products",
            
            # Operational questions
            "SKUs with worst SLA performance",
            "Most marketing-efficient products",
            "Seasonal pattern analysis",
            "Competitive pressure assessment"
        ]
    
    def save_dataset_to_csv(self, output_dir: str = "standard_dataset"):
        """Save the standard dataset to CSV files."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comprehensive dataset
        dataset = self.generate_comprehensive_dataset(weeks=52)
        
        # Save individual SKU files
        for sku_id, data in dataset.items():
            filename = f"{output_dir}/{sku_id}.csv"
            data.to_csv(filename, index=False)
            print(f"✅ Saved {sku_id}: {len(data)} weeks to {filename}")
        
        # Save combined dataset
        combined_data = pd.concat(dataset.values(), ignore_index=True)
        combined_file = f"{output_dir}/all_skus_combined.csv"
        combined_data.to_csv(combined_file, index=False)
        print(f"✅ Saved combined dataset: {len(combined_data)} rows to {combined_file}")
        
        # Save metadata
        metadata = {
            'total_skus': len(dataset),
            'total_weeks': 52,
            'categories': list(set([self.skus[sku]['category'] for sku in dataset.keys()])),
            'scenarios': list(set([data['scenario'].iloc[0] for data in dataset.values()])),
            'date_range': [combined_data['week_date'].min(), combined_data['week_date'].max()],
            'total_gmv': combined_data['gmv'].sum(),
            'total_units': combined_data['units'].sum()
        }
        
        import json
        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Saved metadata to {output_dir}/dataset_metadata.json")
        
        # Save sample queries
        queries = self.create_standard_queries()
        with open(f"{output_dir}/sample_queries.txt", 'w') as f:
            for i, query in enumerate(queries, 1):
                f.write(f"{i}. {query}\n")
        
        print(f"✅ Saved {len(queries)} sample queries to {output_dir}/sample_queries.txt")
        
        return output_dir

# Create and save the standard dataset
if __name__ == "__main__":
    generator = StandardDatasetGenerator()
    output_path = generator.save_dataset_to_csv()
    
    print(f"\n🎉 Standard dataset created successfully!")
    print(f"📁 Location: {output_path}/")
    print(f"📊 12 SKUs across 4 categories")
    print(f"📈 52 weeks of realistic business data")
    print(f"🔍 Multiple business scenarios and patterns")
