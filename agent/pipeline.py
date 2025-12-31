"""
Unified Claims Processing Pipeline
Integrates OCR extraction + fraud detection orchestration.
Entry point for end-to-end claims fraud analysis.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict, is_dataclass
from enum import Enum

from agent.extractor import ocr_pdf
from agent.extractor_enhanced import EnhancedClaimExtractor
from agent.orchestrator_enhanced import EnhancedFraudDetectionOrchestrator, ExecutionMode
from agent.extractors.data_validator import validate_claim_pre_agent


def json_serializable(obj: Any) -> Any:
    """Recursively convert object to JSON-serializable format"""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.name
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: json_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # Fallback: convert to string
    return str(obj)


def process_claim_full_pipeline(
    pdf_path: str,
    execution_mode: str = "parallel",
    historical_data: Optional[list] = None,
    id_document_path: Optional[str] = None,
    save_results: bool = True,
    mode: str = "full_analysis"  # "ocr_only" or "full_analysis"
) -> Dict:
    """
    End-to-end claims processing:
    1. Extract text from PDF using OCR
    2. Extract structured fields from OCR text
    3. Run fraud detection agents on extracted data
    4. Generate unified fraud risk report
    5. Save results to JSON
    
    Args:
        pdf_path: Path to claim PDF
        execution_mode: "parallel", "sequential", or "mixed"
        historical_data: Previous claims for cross-document analysis
        id_document_path: Path to identity document
        save_results: Whether to save JSON outputs
        
    Returns:
        Unified fraud detection report
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(pdf_path)

    # Step 1: OCR extraction
    print(f"[1/3] Extracting text from {p.name}...")
    try:
        ocr_text = ocr_pdf(pdf_path, use_vision_if_available=True)
    except Exception as e:
        print(f"OCR failed: {e}")
        return {"error": f"OCR extraction failed: {e}"}

    # Step 2: Field extraction (enhanced)
    print(f"[2/3] Extracting structured fields...")
    extractor = EnhancedClaimExtractor()
    claim_data = extractor.extract_fields(ocr_text)
    claim_data = {k: (asdict(v) if hasattr(v, '__dataclass_fields__') else v) for k, v in claim_data.items()}
    # DEBUG: Log extracted claim_data for diagnosis
    print(f"[DEBUG] Extracted claim_data for {p.name}:")
    for key, val in claim_data.items():
        print(f"    {key}: {val}")

    if mode == "ocr_only":
        # Only OCR and field extraction, no agent analysis
        result = {
            "ocr_text": ocr_text,
            "claim_data": claim_data,
            "mode": "ocr_only"
        }
        if save_results:
            outputs_dir = Path("Data/outputs")
            outputs_dir.mkdir(parents=True, exist_ok=True)
            claim_out = outputs_dir / f"{p.stem}_claim.json"
            with open(claim_out, "w", encoding="utf-8") as f:
                json.dump(claim_data, f, indent=2, ensure_ascii=False)
        return result

    # Step 2.5: CRITICAL - Pre-agent data validation
    # This filters OCR garbage, validates amounts, removes form template text BEFORE agents see it
    print(f"[2.5/3] Validating and cleaning extracted data...")
    try:
        claim_data, validation_issues = validate_claim_pre_agent(claim_data)
        if validation_issues:
            print(f"    [!] Pre-agent validation found {len(validation_issues)} issues:")
            for issue in validation_issues[:5]:  # Show first 5
                print(f"        - {issue}")
            if len(validation_issues) > 5:
                print(f"        ... and {len(validation_issues) - 5} more")
    except Exception as e:
        print(f"    [!] Pre-agent validation failed: {e} - continuing with original data")

    # Step 3: Fraud detection (full analysis)
    print(f"[3/3] Running fraud detection agents ({execution_mode} mode)...")
    mode_enum = ExecutionMode[execution_mode.upper()]
    orchestrator = EnhancedFraudDetectionOrchestrator(execution_mode=mode_enum)
    fraud_report = orchestrator.analyze(claim_data, historical_data, id_document_path)

    # Run additional agents (ClinicalConsistencyAgent, FinancialReconciliationAgent)
    try:
        from agent.agents_fraud import ClinicalConsistencyAgent, FinancialReconciliationAgent
        clinical_agent = ClinicalConsistencyAgent()
        financial_agent = FinancialReconciliationAgent()
        clinical_result = clinical_agent.analyze(claim_data)
        financial_result = financial_agent.analyze(claim_data)
        # Add to fraud_report
        if "agent_results" not in fraud_report:
            fraud_report["agent_results"] = {}
        fraud_report["agent_results"]["clinical_consistency"] = json_serializable(clinical_result)
        fraud_report["agent_results"]["financial_reconciliation"] = json_serializable(financial_result)
    except Exception as e:
        print(f"Warning: Additional agents failed: {e}")

    # Step 4: Save results
    if save_results:
        outputs_dir = Path("Data/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        # Save claim extraction (UTF-8 encoding for Unicode symbols like ₹)
        claim_out = outputs_dir / f"{p.stem}_claim.json"
        with open(claim_out, "w", encoding="utf-8") as f:
            json.dump(claim_data, f, indent=2, ensure_ascii=False)
        # Save fraud report
        report_out = outputs_dir / f"{p.stem}_fraud_report.json"
        with open(report_out, "w", encoding="utf-8") as f:
            report_to_save = {k: v for k, v in fraud_report.items() if k != "claim_data"}
            # Convert all enums and dataclasses to JSON-serializable format
            report_to_save = json_serializable(report_to_save)
            json.dump(report_to_save, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {outputs_dir}/")
    
    # Ensure returned report is also fully serializable
    return json_serializable(fraud_report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agent.pipeline <pdf-path> [--mode parallel|sequential|mixed]")
    else:
        pdf = sys.argv[1]
        mode = "parallel"
        if "--mode" in sys.argv:
            mode = sys.argv[sys.argv.index("--mode") + 1]

        result = process_claim_full_pipeline(pdf, execution_mode=mode)
        print("\n" + "=" * 60)
        print(f"FRAUD RISK SCORE: {result.get('fraud_risk_score', 'N/A')}/100")
        print(f"OVERALL RISK: {result.get('overall_risk_level', 'N/A')}")
        print(f"RECOMMENDED ACTION: {result.get('recommended_action', 'N/A')}")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
