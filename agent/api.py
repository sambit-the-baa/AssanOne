from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from typing import Optional

from agent.pipeline import process_claim_full_pipeline
from agent.database import ClaimsDatabase
from agent.orchestrator_enhanced import ExecutionMode

app = FastAPI(title="TPA Fraud Detection API", version="2.0")
db = ClaimsDatabase()

logger = logging.getLogger(__name__)

@app.post("/api/claims/analyze")
async def analyze_claim(
    file: UploadFile = File(...),
    execution_mode: str = "parallel",
    user_id: Optional[str] = None,
):
    """
    Analyze single claim and save results
    """
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"Data/uploads/{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        # Process claim
        fraud_report = process_claim_full_pipeline(
            str(temp_path),
            execution_mode=execution_mode,
            save_results=True,
        )
        # Save to database
        claim_id = db.save_claim_result(
            fraud_report.get('claim_data', {}),
            fraud_report,
        )
        # Log audit event
        db.log_audit_event(
            claim_id,
            "CLAIM_ANALYZED",
            user_id=user_id,
            details=f"Risk: {fraud_report.get('overall_risk_level')}",
        )
        return JSONResponse({
            "status": "success",
            "claim_id": claim_id,
            "fraud_risk_score": fraud_report.get("fraud_risk_score"),
            "risk_level": fraud_report.get("overall_risk_level"),
            "recommended_action": fraud_report.get("recommended_action"),
        })
    except Exception as e:
        logger.error(f"Error analyzing claim: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """
    Get real-time dashboard metrics
    """
    try:
        metrics = db.get_dashboard_metrics()
        return JSONResponse(metrics)
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/claims/search")
async def search_claims(
    claim_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    """Search claims by filters (not yet implemented)"""
    return JSONResponse({"status": "not_implemented"})

@app.post("/api/claims/{claim_id}/update-status")
async def update_claim_status(
    claim_id: str,
    new_status: str,
    user_id: Optional[str] = None,
):
    """Update claim status and log audit trail (not yet implemented)"""
    return JSONResponse({"status": "not_implemented"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
