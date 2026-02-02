"""FastAPI application main entry point."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime

from ..agent.graph import MMMAgentWorkflow
from ..data.synthetic import get_sample_queries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MMM Agent System API",
    description="Agentic Marketing Mix & Brand Equity Decision System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent workflow
agent_workflow = MMMAgentWorkflow()

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    user_query: str
    scope: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    session_id: str
    user_query: str
    completed: bool
    confidence_score: float
    error_message: Optional[str] = None
    timestamp: str
    kpi_summary: Optional[Dict[str, Any]] = None
    mmm_decomposition: Optional[Dict[str, Any]] = None
    brand_equity: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    scenarios: Optional[List[Dict[str, Any]]] = None
    explainability: Optional[Dict[str, Any]] = None
    hitl_required: Optional[bool] = None
    hitl_status: Optional[str] = None
    human_decision: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class SampleQueriesResponse(BaseModel):
    queries: List[str]

class WorkflowStateResponse(BaseModel):
    session_id: str
    state: Optional[Dict[str, Any]] = None


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MMM Agent System API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_query(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Run MMM analysis on user query.
    
    This endpoint processes natural language queries about marketing mix modeling,
    brand equity, and business decision intelligence.
    """
    try:
        logger.info(f"Received analysis request: {request.user_query}")
        
        # Run the analysis
        result = agent_workflow.run_analysis(request.user_query, request.scope)
        
        # Convert to response format
        response = AnalysisResponse(**result)
        
        # Log completion in background
        background_tasks.add_task(
            log_analysis_completion, 
            result.get("session_id"), 
            request.user_query,
            result.get("completed", False)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample-queries", response_model=SampleQueriesResponse)
async def get_sample_queries():
    """Get sample queries for testing."""
    try:
        queries = get_sample_queries()
        return SampleQueriesResponse(queries=queries)
    except Exception as e:
        logger.error(f"Error getting sample queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/{session_id}/state", response_model=WorkflowStateResponse)
async def get_workflow_state(session_id: str):
    """Get the current state of a workflow session."""
    try:
        state = agent_workflow.get_workflow_state(session_id)
        return WorkflowStateResponse(session_id=session_id, state=state)
    except Exception as e:
        logger.error(f"Error getting workflow state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/visualize")
async def visualize_workflow():
    """Get a text visualization of the workflow."""
    try:
        visualization = agent_workflow.visualize_workflow()
        return {"workflow": visualization}
    except Exception as e:
        logger.error(f"Error visualizing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hitl/{session_id}/approve")
async def approve_hitl_action(session_id: str, approval_data: Dict[str, Any]):
    """
    Approve a HITL action (for testing/demo purposes).
    
    In production, this would be integrated with proper authentication
    and approval workflows.
    """
    try:
        # This is a simplified implementation for demo
        # In production, you'd update the actual workflow state
        
        logger.info(f"HITL approval for session {session_id}: {approval_data}")
        
        return {
            "session_id": session_id,
            "status": "approved",
            "message": "Action approved successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in HITL approval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hitl/{session_id}/reject")
async def reject_hitl_action(session_id: str, rejection_data: Dict[str, Any]):
    """
    Reject a HITL action (for testing/demo purposes).
    """
    try:
        logger.info(f"HITL rejection for session {session_id}: {rejection_data}")
        
        return {
            "session_id": session_id,
            "status": "rejected",
            "message": "Action rejected successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in HITL rejection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/skus")
async def get_available_skus():
    """Get list of available SKUs for analysis."""
    try:
        from ..data.synthetic import data_generator
        skus = list(data_generator.skus.keys())
        
        sku_info = {}
        for sku in skus:
            info = data_generator.get_sku_info(sku)
            sku_info[sku] = {
                "name": info.get("name", "Unknown"),
                "base_price": info.get("base_price", 0),
                "base_weekly_demand": info.get("base_weekly_demand", 0)
            }
        
        return {"skus": sku_info}
    except Exception as e:
        logger.error(f"Error getting SKUs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for logging
async def log_analysis_completion(session_id: str, query: str, completed: bool):
    """Log analysis completion for monitoring."""
    status = "completed" if completed else "failed"
    logger.info(f"Analysis {status} - Session: {session_id}, Query: {query}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("MMM Agent System API starting up...")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set. Some features may not work.")
    
    logger.info("MMM Agent System API ready!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("MMM Agent System API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
