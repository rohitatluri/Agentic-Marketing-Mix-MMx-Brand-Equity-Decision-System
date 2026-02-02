"""
Use Standard Dataset with MMM Agent System
"""

import pandas as pd
from app.agent.graph import MMMAgentWorkflow
import app.data.synthetic as synthetic

def load_standard_dataset():
    """Load the standard dataset and replace demo data."""
    
    # Load the combined dataset
    combined_data = pd.read_csv("standard_dataset/all_skus_combined.csv")
    
    # Convert date column
    combined_data['week_date'] = pd.to_datetime(combined_data['week_date'])
    
    # Group by SKU to create the expected format
    sku_data = {}
    for sku_id in combined_data['sku_id'].unique():
        sku_data[sku_id] = combined_data[combined_data['sku_id'] == sku_id].copy()
    
    # Replace the demo data
    synthetic.demo_data = sku_data
    
    print(f"✅ Loaded standard dataset with {len(sku_data)} SKUs")
    print(f"📊 Date range: {combined_data['week_date'].min()} to {combined_data['week_date'].max()}")
    print(f"💰 Total GMV: ${combined_data['gmv'].sum():,.0f}")
    print(f"📦 Total Units: {combined_data['units'].sum():,}")
    
    return sku_data

def test_standard_dataset():
    """Test the MMM Agent with the standard dataset."""
    
    print("🚀 Testing MMM Agent with Standard Dataset\n")
    
    # Load the standard dataset
    load_standard_dataset()
    
    # Initialize the workflow
    workflow = MMMAgentWorkflow()
    
    # Test queries from the standard dataset
    test_queries = [
        "Analyze overall performance across all categories",
        "Which SKUs are showing the strongest growth trend?",
        "Why are discount-heavy SKUs underperforming?",
        "What if we reduce discounts by 10% across all SKUs?",
        "Compare Electronics vs Fashion category performance"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 TEST {i}: {query}")
        print(f"{'='*60}")
        
        try:
            result = workflow.run_analysis(query)
            
            print(f"✅ Status: {'Completed' if result['completed'] else 'Failed'}")
            print(f"📈 Confidence: {result['confidence_score']:.1f}%")
            print(f"⏱️  Processing time: {result.get('processing_time', 'N/A')}")
            
            if result.get('error_message'):
                print(f"❌ Error: {result['error_message']}")
            
            results.append({
                'query': query,
                'completed': result['completed'],
                'confidence': result['confidence_score'],
                'error': result.get('error_message')
            })
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({
                'query': query,
                'completed': False,
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    completed = sum(1 for r in results if r['completed'])
    total = len(results)
    avg_confidence = sum(r['confidence'] for r in results) / total
    
    print(f"✅ Completed: {completed}/{total} ({completed/total*100:.1f}%)")
    print(f"📈 Average Confidence: {avg_confidence:.1f}%")
    
    print(f"\n🎯 Detailed Results:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['completed'] else "❌"
        print(f"   {i}. {status} {r['confidence']:.1f}% - {r['query'][:50]}...")
    
    return results

if __name__ == "__main__":
    test_standard_dataset()
