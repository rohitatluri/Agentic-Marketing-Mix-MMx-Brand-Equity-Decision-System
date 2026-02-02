#!/usr/bin/env python3
"""
Demo script for MMM Agent System

This script demonstrates the core functionality of the MMM Agent System
by running sample queries and showing the results.
"""

import os
import sys
import asyncio
from datetime import datetime
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.agent.graph import MMMAgentWorkflow
from app.data.synthetic import get_sample_queries


def print_separator(title):
    """Print a formatted separator."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_result(result):
    """Print analysis result in a formatted way."""
    print(f"\n📊 ANALYSIS RESULTS")
    print(f"Session ID: {result.get('session_id', 'N/A')}")
    print(f"Completed: {'✅' if result.get('completed', False) else '❌'}")
    print(f"Confidence: {result.get('confidence_score', 0):.1%}")
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    
    if result.get('error_message'):
        print(f"❌ Error: {result['error_message']}")
        return
    
    # KPI Summary
    if result.get('kpi_summary'):
        kpi = result['kpi_summary']
        print(f"\n📈 KPI SUMMARY:")
        print(f"   GMV Change: {kpi.get('gmv_change_pct', 0):.1f}%")
        print(f"   Units Change: {kpi.get('units_change_pct', 0):.1f}%")
        print(f"   Data Quality: {kpi.get('data_quality_score', 0):.1%}")
    
    # MMM Decomposition
    if result.get('mmm_decomposition'):
        mmm = result['mmm_decomposition']
        print(f"\n🔍 DRIVER ATTRIBUTION:")
        print(f"   Total Change: {mmm.get('total_change_pct', 0):.1f}%")
        print(f"   Model Fit (R²): {mmm.get('model_fit_r2', 0):.3f}")
        
        if mmm.get('driver_contributions'):
            print("   Top Drivers:")
            for driver in mmm['driver_contributions'][:3]:
                print(f"     • {driver.get('driver', 'Unknown')}: {driver.get('contribution_pct', 0):.1f}%")
    
    # Brand Equity
    if result.get('brand_equity'):
        brand = result['brand_equity']
        print(f"\n🏆 BRAND EQUITY:")
        print(f"   Brand Index: {brand.get('brand_equity_index', 0):.1f}/100")
        print(f"   Trend: {brand.get('trend_direction', 'Unknown')}")
        print(f"   Confidence: {brand.get('confidence_level', 0):.1%}")
    
    # Recommendations
    if result.get('recommendations'):
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"   {i}. {rec.get('action', 'No action')}")
            print(f"      Short-term: {rec.get('short_term_impact', 'N/A')}")
            print(f"      Long-term: {rec.get('long_term_impact', 'N/A')}")
            print(f"      Risk: {rec.get('risk_level', 'Unknown')}")
            print(f"      Confidence: {rec.get('confidence_score', 0):.1%}")
    
    # HITL Information
    if result.get('hitl_required'):
        print(f"\n⚠️  HUMAN-IN-THE-LOOP REQUIRED")
        print(f"   Status: {result.get('hitl_status', 'Unknown')}")
        if result.get('human_decision'):
            decision = result['human_decision']
            print(f"   Decision: {decision.get('decision', 'Unknown')}")
            print(f"   Approver: {decision.get('approver', 'Unknown')}")


async def run_demo():
    """Run the demo with sample queries."""
    print_separator("🚀 MMM AGENT SYSTEM DEMO")
    
    # Initialize the workflow
    print("Initializing MMM Agent Workflow...")
    workflow = MMMAgentWorkflow()
    
    # Get sample queries
    sample_queries = get_sample_queries()
    
    print(f"Found {len(sample_queries)} sample queries")
    print("\nAvailable queries:")
    for i, query in enumerate(sample_queries, 1):
        print(f"   {i}. {query}")
    
    # Run a few sample queries
    demo_queries = sample_queries[:3]  # Run first 3 queries
    
    for i, query in enumerate(demo_queries, 1):
        print_separator(f"DEMO QUERY {i}: {query}")
        
        print(f"🤖 Processing: {query}")
        start_time = datetime.now()
        
        try:
            # Run the analysis
            result = workflow.run_analysis(query)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print_result(result)
            
        except Exception as e:
            print(f"❌ Error running query: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print_separator("🎉 DEMO COMPLETED")
    print("The MMM Agent System demo has completed successfully!")
    print("\nTo explore further:")
    print("1. Run the FastAPI server: python -m app.api.main")
    print("2. Open http://localhost:8000/docs for API documentation")
    print("3. Try different queries using the /analyze endpoint")
    print("4. Check the workflow visualization at /workflow/visualize")


def test_individual_components():
    """Test individual components of the system."""
    print_separator("🧪 COMPONENT TESTING")
    
    try:
        # Test data generator
        print("Testing data generator...")
        from app.data.synthetic import data_generator, demo_data
        
        print(f"✅ Generated data for {len(demo_data)} SKUs")
        for sku_id, data in demo_data.items():
            print(f"   {sku_id}: {len(data)} weeks of data")
        
        # Test tools
        print("\nTesting KPI tools...")
        from app.tools.kpi_tools import KPITools
        from app.tools.schemas import TimeRange
        from datetime import date
        
        kpi_tools = KPITools()
        time_range = TimeRange(
            start_date=date.today() - timedelta(weeks=4),
            end_date=date.today()
        )
        
        kpi_summary = kpi_tools.get_kpi_summary(["SKU-123"], time_range)
        print(f"✅ KPI Summary: GMV change {kpi_summary.gmv_change_pct:.1f}%")
        
        # Test MMM tools
        print("\nTesting MMM tools...")
        from app.tools.mmm_tools import MMMTools
        
        mmm_tools = MMMTools()
        mmm_decomp = mmm_tools.get_mmm_decomposition(["SKU-123"], time_range)
        print(f"✅ MMM Decomposition: R² = {mmm_decomp.model_fit_r2:.3f}")
        
        # Test brand equity tools
        print("\nTesting brand equity tools...")
        from app.tools.brand_equity_tools import BrandEquityTools
        
        brand_tools = BrandEquityTools()
        brand_tools.set_mmm_tools(mmm_tools)
        brand_metrics = brand_tools.get_brand_equity(["SKU-123"], time_range)
        print(f"✅ Brand Equity: Index {brand_metrics.brand_equity_index:.1f}")
        
        print("\n✅ All components working correctly!")
        
    except Exception as e:
        print(f"❌ Component testing failed: {str(e)}")
        import traceback
        traceback.print_exc()


def show_system_info():
    """Show system information and setup instructions."""
    print_separator("ℹ️  SYSTEM INFORMATION")
    
    print("MMM Agent System - Agentic Marketing Mix & Brand Equity Decision System")
    print("\n🏗️  Architecture:")
    print("   • LangGraph workflow for agent orchestration")
    print("   • Pydantic schemas for type safety")
    print("   • FastAPI for REST API")
    print("   • Synthetic data generation for demo")
    print("   • Human-in-the-Loop (HITL) for governance")
    
    print("\n📁 Project Structure:")
    print("   app/")
    print("   ├── agent/          # LangGraph workflow")
    print("   ├── tools/          # Analysis tools")
    print("   ├── models/         # MMM & Brand models")
    print("   ├── api/            # FastAPI application")
    print("   └── data/           # Synthetic data")
    
    print("\n🚀 Quick Start:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Set OpenAI API key: export OPENAI_API_KEY=your_key")
    print("   3. Run demo: python demo.py")
    print("   4. Start API: python -m app.api.main")
    print("   5. Open docs: http://localhost:8000/docs")
    
    print("\n🔑 Environment Variables:")
    print("   • OPENAI_API_KEY: Required for LLM functionality")
    print("   • LOG_LEVEL: Logging level (INFO, DEBUG, etc.)")


if __name__ == "__main__":
    print("🎯 MMM Agent System Demo")
    print("Choose an option:")
    print("1. Run full demo")
    print("2. Test individual components")
    print("3. Show system information")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            asyncio.run(run_demo())
        elif choice == "2":
            test_individual_components()
        elif choice == "3":
            show_system_info()
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please run again.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
