#!/usr/bin/env python3
"""
Streamlit UI for MMM Agent System Testing

A simple web interface to test the MMM Agent System with sample queries
and visualize results.
"""

import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import time

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure page
st.set_page_config(
    page_title="MMM Agent System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    color: #000000;
}
.result-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e1e5e9;
    margin: 10px 0;
    color: #000000;
}
.result-card h4 {
    color: #1f2937;
    margin-bottom: 10px;
}
.result-card p {
    color: #374151;
    margin-bottom: 8px;
}
.error-card {
    background-color: #fff5f5;
    border: 1px solid #fed7d7;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def load_agent():
    """Load the MMM Agent Workflow."""
    try:
        from app.agent.graph import MMMAgentWorkflow
        return MMMAgentWorkflow()
    except Exception as e:
        st.error(f"Error loading agent: {str(e)}")
        return None

def load_sample_queries():
    """Load sample queries for testing."""
    try:
        from app.data.synthetic import get_sample_queries
        return get_sample_queries()
    except Exception as e:
        st.error(f"Error loading sample queries: {str(e)}")
        return []

def display_results(result):
    """Display analysis results in a formatted way."""
    if result.get('error_message'):
        st.markdown(f"""
        <div class="error-card">
            <h4>❌ Analysis Error</h4>
            <p><strong>Error:</strong> {result['error_message']}</p>
            <p><strong>Session ID:</strong> {result.get('session_id', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Status", 
            "✅ Complete" if result.get('completed', False) else "❌ Failed",
            delta=None
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{result.get('confidence_score', 0):.0%}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Analysis Time",
            f"{result.get('processing_time', 0):.2f}s",
            delta=None
        )
    
    with col4:
        st.metric(
            "Session ID",
            result.get('session_id', 'N/A')[:8] + "...",
            delta=None
        )
    
    # KPI Summary
    if result.get('kpi_summary'):
        st.markdown("### 📈 KPI Summary")
        kpi = result['kpi_summary']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("GMV Change", f"{kpi.get('gmv_change_pct', 0):.1f}%")
        with col2:
            st.metric("Units Change", f"{kpi.get('units_change_pct', 0):.1f}%")
        with col3:
            st.metric("Data Quality", f"{kpi.get('data_quality_score', 0):.1%}")
    
    # MMM Decomposition
    if result.get('mmm_decomposition'):
        st.markdown("### 🔍 Driver Attribution")
        mmm = result['mmm_decomposition']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Change", f"{mmm.get('total_change_pct', 0):.1f}%")
        with col2:
            st.metric("Model Fit (R²)", f"{mmm.get('model_fit_r2', 0):.3f}")
        
        if mmm.get('driver_contributions'):
            st.markdown("**Top Drivers:**")
            for driver in mmm['driver_contributions'][:5]:
                contribution = driver.get('contribution_pct', 0)
                color = "🔴" if contribution < -2 else "🟢" if contribution > 2 else "🟡"
                st.write(f"{color} **{driver.get('driver', 'Unknown')}**: {contribution:.1f}%")
    
    # Brand Equity
    if result.get('brand_equity'):
        st.markdown("### 🏆 Brand Equity")
        brand = result['brand_equity']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Brand Index", f"{brand.get('brand_equity_index', 0):.1f}/100")
        with col2:
            st.metric("Trend", brand.get('trend_direction', 'Unknown'))
        with col3:
            st.metric("Confidence", f"{brand.get('confidence_level', 0):.1%}")
    
    # Recommendations
    if result.get('recommendations'):
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            risk_color = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🔴'
            }.get(rec.get('risk_level', 'unknown'), '⚪')
            
            # Use st.expander for better visibility
            with st.expander(f"{i}. {rec.get('action', 'No action')} {risk_color}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Short-term Impact:** {rec.get('short_term_impact', 'N/A')}")
                    st.markdown(f"**Long-term Impact:** {rec.get('long_term_impact', 'N/A')}")
                with col2:
                    st.markdown(f"**Risk Level:** {risk_color} {rec.get('risk_level', 'Unknown').title()}")
                    st.markdown(f"**Confidence:** {rec.get('confidence_score', 0):.1%}")
                
                if rec.get('caveats'):
                    st.warning("⚠️ " + "; ".join(rec['caveats']))
    
    # Scenarios
    if result.get('scenarios'):
        st.markdown("### 🎯 Scenario Analysis")
        for i, scenario in enumerate(result['scenarios'][:2], 1):
            with st.expander(f"Scenario {i}"):
                # Interventions
                st.markdown("**Interventions:**")
                for intervention in scenario.get('interventions', []):
                    st.markdown(f"- {intervention.get('parameter', 'Unknown')}: {intervention.get('change', 0)}{intervention.get('unit', '')}")
                
                # Impact
                impact = scenario.get('impact', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Short-term GMV Impact", f"{impact.get('short_term_gmv_impact', 0):.1f}%")
                    st.metric("Net ROI", f"{impact.get('net_roi', 0):.1f}")
                with col2:
                    st.metric("Long-term Brand Impact", f"{impact.get('long_term_brand_impact', 0):.1f} points")
                    st.metric("Confidence", f"{impact.get('confidence_score', 0):.1%}")
                
                risk_level = impact.get('risk_level', 'unknown')
                if risk_level == 'high':
                    st.error(f"Risk Level: {risk_level.title()}")
                elif risk_level == 'medium':
                    st.warning(f"Risk Level: {risk_level.title()}")
                else:
                    st.success(f"Risk Level: {risk_level.title()}")
    
    # HITL Information
    if result.get('hitl_required'):
        st.markdown("### ⚠️ Human-in-the-Loop Required")
        st.warning(f"HITL Status: {result.get('hitl_status', 'Unknown')}")
        if result.get('human_decision'):
            decision = result['human_decision']
            st.info(f"Decision: {decision.get('decision', 'Unknown')} by {decision.get('approver', 'Unknown')}")

def main():
    """Main Streamlit application."""
    st.title("📊 MMM Agent System")
    st.markdown("**Agentic Marketing Mix & Brand Equity Decision System**")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Load agent
    agent = load_agent()
    if not agent:
        st.error("Failed to load MMM Agent. Please check your configuration.")
        return
    
    # Sample queries
    sample_queries = load_sample_queries()
    
    # Query input section
    st.sidebar.markdown("### 🎯 Test Query")
    
    # Option to select sample query or enter custom
    query_option = st.sidebar.radio(
        "Choose query type:",
        ["Sample Queries", "Custom Query"]
    )
    
    user_query = ""
    if query_option == "Sample Queries" and sample_queries:
        selected_query = st.sidebar.selectbox(
            "Select a sample query:",
            sample_queries,
            index=0
        )
        user_query = selected_query
    else:
        user_query = st.sidebar.text_area(
            "Enter your query:",
            placeholder="e.g., Why did GMV drop for SKU-123 in the last 4 weeks?",
            height=100
        )
    
    # SKU scope selection
    st.sidebar.markdown("### 📦 SKU Scope")
    sku_options = ["All SKUs", "SKU-123", "SKU-456", "SKU-789"]
    selected_skus = st.sidebar.multiselect(
        "Select SKUs to analyze:",
        sku_options,
        default=["All SKUs"]
    )
    
    # Convert SKU selection
    scope = None
    if "All SKUs" not in selected_skus:
        scope = selected_skus
    
    # Analyze button
    analyze_button = st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    if analyze_button and user_query:
        with st.spinner("🤖 Running MMM Agent Analysis..."):
            start_time = time.time()
            
            try:
                result = agent.run_analysis(user_query, scope)
                processing_time = time.time() - start_time
                result['processing_time'] = processing_time
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'query': user_query,
                    'result': result
                })
                
                # Display results
                st.markdown(f"## 📊 Analysis Results")
                st.markdown(f"**Query:** {user_query}")
                display_results(result)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # Quick start section
    if not st.session_state.analysis_history:
        st.markdown("""
        ## 🚀 Quick Start
        
        1. **Select a sample query** from the sidebar or enter your own
        2. **Choose SKU scope** (All SKUs or specific ones)
        3. **Click "Run Analysis"** to see the agent in action
        
        ### 🎯 Sample Questions to Try:
        """)
        
        if sample_queries:
            for i, query in enumerate(sample_queries[:5], 1):
                st.markdown(f"{i}. **{query}**")
        
        st.markdown("""
        ### 📋 What the Agent Does:
        - 🔍 **Root Cause Analysis**: Identifies what drives GMV changes
        - 🏆 **Brand Equity Tracking**: Monitors brand health impact
        - 🎯 **Scenario Simulation**: Tests "what-if" interventions
        - 🛡️ **Risk Assessment**: Flags high-risk actions for human review
        - 💡 **Recommendations**: Provides actionable next steps
        """)
    
    # Analysis history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("## 📜 Analysis History")
        
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
            with st.expander(f"{i}. {analysis['query'][:50]}... - {analysis['timestamp'].strftime('%H:%M:%S')}"):
                display_results(analysis['result'])
    
    # System information
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        **MMM Agent System Features:**
        - 🤖 LangGraph-based agent workflow
        - 📊 Marketing Mix Model decomposition
        - 🏆 Brand Equity Index tracking
        - 🎯 Scenario simulation with confidence scoring
        - 🛡️ Human-in-the-Loop governance
        - 📋 Complete explainability and audit trail
        
        **Test Scenarios:**
        - **SKU-123**: SLA degradation scenario
        - **SKU-456**: Procurement issues scenario  
        - **SKU-789**: Marketing campaign scenario
        """)

if __name__ == "__main__":
    main()
