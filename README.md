# 🎯 MMM Agent System - Agentic Marketing Mix & Brand Equity Intelligence

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Agentic Approach](#agentic-approach)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [File Structure](#file-structure)

---

## 🎯 Problem Statement

Traditional marketing mix modeling (MMM) and brand equity analysis face several critical challenges:

1. **Complex Analysis Requirements**: MMM requires sophisticated statistical modeling and domain expertise
2. **Data Integration Complexity**: Combining multiple data sources (sales, marketing, operations) is difficult
3. **Real-time Decision Making**: Businesses need quick insights, not lengthy analysis cycles
4. **Scenario Planning**: Testing "what-if" scenarios requires manual recalibration
5. **Risk Assessment**: Identifying high-risk recommendations needs human judgment
6. **Explainability**: Understanding why the model makes certain recommendations is crucial

**Key Question**: How can we automate complex marketing analytics while maintaining accuracy, explainability, and human oversight?

---

## 🚀 Solution Overview

We've built an **Agentic MMM System** that uses LangGraph to create an intelligent workflow that:

- **Automatically analyzes** marketing performance across multiple dimensions
- **Generates actionable recommendations** with confidence scores
- **Simulates scenarios** to test business decisions
- **Provides explainable insights** with step-by-step reasoning
- **Includes human-in-the-loop** for high-risk decisions

### 🤖 The Agentic Advantage

Unlike traditional analytics tools, our system uses multiple specialized AI agents that work together:

1. **Data Validation Agent**: Ensures data quality and completeness
2. **Planning Agent**: Determines analysis strategy based on query type
3. **Analysis Agent**: Executes KPI, MMM, and Brand Equity analysis
4. **Diagnosis Agent**: Identifies root causes and patterns
5. **Scenario Agent**: Simulates business interventions
6. **Risk Assessment Agent**: Evaluates recommendation risks
7. **Recommendation Agent**: Ranks and prioritizes actions
8. **Explainability Agent**: Provides audit trails and reasoning

---

## 🧠 Agentic Approach

### Workflow Orchestration

The system uses **LangGraph** to create a sophisticated workflow that processes user queries through multiple specialized nodes:

```
User Query → Intake → Validation → Planning → Analysis → Diagnosis → Scenarios → Risk Assessment → Recommendations → Explainability → Completion
```

### Key Agentic Features

#### 1. **Dynamic Planning**
- Analyzes query intent and determines required tools
- Estimates confidence levels based on data availability
- Adapts strategy based on data quality

#### 2. **Multi-Tool Coordination**
- Coordinates KPI analysis, MMM decomposition, and brand equity
- Handles tool failures gracefully with fallback methods
- Manages data dependencies between tools

#### 3. **Human-in-the-Loop (HITL)**
- Automatically flags high-risk recommendations
- Provides approval/rejection workflow
- Maintains audit trails for compliance

#### 4. **Explainability**
- Tracks every decision step
- Provides confidence breakdowns
- Sources data attribution for claims

---

## 🏗️ Architecture

### Core Components

```
├── app/
│   ├── agent/
│   │   ├── graph.py          # LangGraph workflow orchestration
│   │   └── nodes.py          # Individual agent nodes
│   ├── tools/
│   │   ├── kpi_tools.py      # KPI analysis tools
│   │   ├── mmm_tools.py      # Marketing Mix Modeling
│   │   ├── brand_equity_tools.py  # Brand equity analysis
│   │   ├── scenario_tools.py # Scenario simulation
│   │   └── schemas.py        # Data models and validation
│   ├── models/
│   │   ├── mmm_model.py      # MMM statistical models
│   │   └── brand_equity.py   # Brand equity calculations
│   ├── data/
│   │   └── synthetic.py      # Synthetic data generator
│   └── api/
│       └── main.py           # FastAPI REST API
├── streamlit_app.py          # Web UI
├── demo.py                   # Demo script
└── standard_dataset/         # Comprehensive test data
```

### Technology Stack

- **Orchestration**: LangGraph for workflow management
- **LLM**: OpenAI GPT for reasoning and analysis
- **Backend**: FastAPI for REST API
- **Frontend**: Streamlit for interactive UI
- **Analytics**: Pandas, NumPy, SciPy for data processing
- **ML**: Scikit-learn for statistical modeling

---

## 📊 Dataset

### Synthetic Data Generation

We've created a comprehensive synthetic dataset that mimics real-world e-commerce data:

#### **Dataset Characteristics**
- **12 SKUs** across **4 categories** (Electronics, Fashion, Home, Sports)
- **52 weeks** of historical data
- **Multiple business scenarios** (growth, decline, volatility, etc.)
- **Realistic patterns** with seasonality and trends

#### **Data Schema**
```python
{
    'week_date': '2025-02-02',        # Week ending date
    'sku_id': 'ELEC-001',             # Product identifier
    'gmv': 150000.0,                  # Gross Merchandise Value
    'units': 125,                     # Units sold
    'price': 1200.0,                  # Base price
    'discount_pct': 10.0,              # Discount percentage
    'sla_days': 1.5,                  # Service Level Agreement
    'marketing_spend': 5000.0,         # Marketing investment
    'stockout_rate': 0.05,             # Stockout frequency
    'week_number': 1,                  # Week number (1-52)
    'category': 'Electronics',         # Product category
    'scenario': 'mixed_performance'   # Business scenario type
}
```

#### **Business Scenarios**
Each SKU follows a unique business pattern:
- **ELEC-001**: Mixed performance with seasonal trends
- **FASH-002**: SLA issues affecting demand
- **SPORT-001**: Volatile demand patterns
- **HOME-003**: Discount-heavy dependency
- And 8 more realistic scenarios...

### Data Usage

The dataset is used to:
1. **Train and test** the MMM models
2. **Demonstrate** different analysis scenarios
3. **Validate** agent recommendations
4. **Showcase** system capabilities

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd MMix_Agents
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy .env.example to .env and add your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_api_key_here
```

---

## 💻 Usage

### Quick Start

1. **Run the demo**
```bash
python demo.py
```

2. **Launch the web UI**
```bash
streamlit run streamlit_app.py
```

3. **Start the API server**
```bash
python -m app.api.main
```

### Sample Queries

The system can handle various types of business questions:

#### **Root Cause Analysis**
- "Why did GMV drop for SKU-123 in the last 4 weeks?"
- "What's driving the brand equity decline for premium products?"

#### **Performance Analysis**
- "Which SKUs are showing the strongest growth trend?"
- "Compare Electronics vs Fashion category performance"

#### **Scenario Planning**
- "What if we reduce discount by 5% but improve SLA by 1 day?"
- "Should we increase marketing spend by 20%?"

#### **Strategic Insights**
- "What are the top 3 recommendations for growth?"
- "How to reduce discount dependency?"

---

## 🧪 Testing

### Running Tests

1. **Basic functionality test**
```bash
python demo.py
# Choose option 1 for full demo
```

2. **Standard dataset test**
```bash
python use_standard_dataset.py
```

### Test Questions

See [TESTING_QUESTIONS.md](TESTING_QUESTIONS.md) for comprehensive test scenarios including:
- Root cause analysis questions
- Performance comparison queries
- Scenario simulation tests
- Edge cases and error handling

---

## 📚 API Documentation

### REST API Endpoints

Start the API server:
```bash
python -m app.api.main
```

Access interactive docs at: `http://localhost:8000/docs`

#### Key Endpoints

- **POST /analyze**: Run MMM analysis
- **GET /workflow/{session_id}/state**: Get workflow state
- **POST /hitl/{session_id}/approve**: Approve recommendations
- **GET /skus**: List available SKUs

---

## 📁 File Structure

```
MMix_Agents/
├── 📄 README.md                    # This file
├── 📄 TESTING_QUESTIONS.md         # Test scenarios
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env                         # Environment variables
├── 🚀 demo.py                     # Demo script
├── 🌐 streamlit_app.py            # Web UI
├── 📊 use_standard_dataset.py     # Dataset testing
├── 📈 standard_dataset.py         # Dataset generator
├── 📁 standard_dataset/           # Generated test data
├── 📁 app/                        # Core application
│   ├── 🤖 agent/
│   │   ├── graph.py               # LangGraph workflow
│   │   └── nodes.py               # Agent implementations
│   ├── 🔧 tools/
│   │   ├── kpi_tools.py           # KPI analysis
│   │   ├── mmm_tools.py           # MMM modeling
│   │   ├── brand_equity_tools.py  # Brand equity
│   │   ├── scenario_tools.py      # Scenario simulation
│   │   └── schemas.py             # Data models
│   ├── 📈 models/
│   │   ├── mmm_model.py           # MMM algorithms
│   │   └── brand_equity.py        # Brand calculations
│   ├── 📊 data/
│   │   └── synthetic.py            # Data generation
│   └── 🌐 api/
│       └── main.py                # FastAPI server
└── 📁 tests/                      # Unit tests
```

---

## 🎯 User Output Examples

### Analysis Results

When you run a query like *"What if we reduce discount by 5% but improve SLA by 1 day?"*, the system provides:

#### 📈 **KPI Summary**
```
GMV Change: 0.0%
Units Change: 0.0%
Data Quality: 100.0%
```

#### 🔍 **Driver Attribution**
```
Total Change: -4.1%
Model Fit (R²): 1.000
Top Drivers:
  • Price/Discount: 47.1%
  • SLA: -124.1%
  • Procurement: -0.8%
```

#### 🏆 **Brand Equity**
```
Brand Index: 5.5/100
Trend: STABLE
Confidence: 85.0%
```

#### 💡 **Recommendations**
```
1. discount 5.0%, sla 1.0days
   Short-term: GMV impact: 241.1%
   Long-term: Brand impact: -0.50 points
   Risk: medium
   Confidence: 90.0%
```

### Agentic Steps Breakdown

The system follows these transparent steps:

1. **🔍 Intake**: Parse query, extract entities (SKUs, time ranges)
2. **✅ Validation**: Check data quality and completeness
3. **📋 Planning**: Determine analysis strategy and required tools
4. **📊 Analysis**: Execute KPI, MMM, and Brand Equity analysis
5. **🔬 Diagnosis**: Identify root causes and patterns
6. **🎯 Scenarios**: Simulate business interventions
7. **⚠️ Risk Assessment**: Evaluate recommendation risks
8. **💡 Recommendations**: Generate ranked action items
9. **📝 Explainability**: Provide audit trail and reasoning
10. **✅ Completion**: Final quality check and summary

Each step includes:
- **Confidence scores**
- **Data sources used**
- **Assumptions made**
- **Potential limitations**

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎉 Key Achievements

✅ **Automated Complex Analytics**: Multi-tool coordination without manual intervention  
✅ **Explainable AI**: Full audit trail and step-by-step reasoning  
✅ **Adaptive Analysis**: Handles data limitations gracefully  
✅ **Risk-Aware**: Human-in-the-loop for critical decisions  
✅ **Real-Time Insights**: Fast response times for business decisions  
✅ **Comprehensive Testing**: Extensive validation with realistic scenarios  

**🚀 Ready for Production**: This system demonstrates how agentic AI can transform complex business analytics into accessible, actionable intelligence.
