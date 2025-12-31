"""
Fraud Detection Orchestration Engine
Coordinates all fraud detection agents and produces unified risk scoring.
Supports parallel and sequential execution modes.
"""
import json
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from enum import Enum

from agent.agents_fraud import (
    OverbillingProtectionAgent,
    FraudDiagnosticAgent,
    UnbundlingUpccodingAgent,
    IdentityTheftProtectionAgent,
    AgentResult,
    RiskLevel
)


class ExecutionMode(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    MIXED = "mixed"  # Critical agents sequential, others parallel


class FraudDetectionOrchestrator:
    """
    Orchestrates all fraud detection agents.
    Produces unified risk scores and remediation recommendations.
    """

    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.PARALLEL):
        self.execution_mode = execution_mode
        self.agents = {
            "overbilling": OverbillingProtectionAgent(),
            "diagnostic": FraudDiagnosticAgent(),
            "unbundling": UnbundlingUpccodingAgent(),
            "identity": IdentityTheftProtectionAgent(),
        }

    def analyze(
        self,
        claim_data: Dict,
        historical_data: Optional[List[Dict]] = None,
        id_document_path: Optional[str] = None
    ) -> Dict:
        """
        Run all fraud detection agents on a claim.
        
        Args:
            claim_data: Extracted claim fields from OCR agent
            historical_data: Previous claims for cross-document analysis
            id_document_path: Path to identity document image
            
        Returns:
            Unified fraud detection report
        """
        results = {}

        if self.execution_mode == ExecutionMode.PARALLEL:
            results = self._run_parallel(claim_data, historical_data, id_document_path)
        elif self.execution_mode == ExecutionMode.SEQUENTIAL:
            results = self._run_sequential(claim_data, historical_data, id_document_path)
        else:  # MIXED
            results = self._run_mixed(claim_data, historical_data, id_document_path)

        # Compute unified score
        unified_report = self._compute_unified_score(results, claim_data)
        return unified_report

    def _run_parallel(self, claim_data: Dict, historical_data: Optional[List[Dict]], id_path: Optional[str]) -> Dict:
        """Run all agents in parallel using ThreadPoolExecutor."""
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.agents["overbilling"].analyze, claim_data): "overbilling",
                executor.submit(self.agents["diagnostic"].analyze, claim_data, historical_data): "diagnostic",
                executor.submit(self.agents["unbundling"].analyze, claim_data): "unbundling",
                executor.submit(self.agents["identity"].analyze, claim_data, id_path): "identity",
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    print(f"Error in {agent_name} agent: {e}")
                    results[agent_name] = None
        return results

    def _run_sequential(self, claim_data: Dict, historical_data: Optional[List[Dict]], id_path: Optional[str]) -> Dict:
        """Run all agents sequentially."""
        results = {}
        try:
            results["overbilling"] = self.agents["overbilling"].analyze(claim_data)
            results["diagnostic"] = self.agents["diagnostic"].analyze(claim_data, historical_data)
            results["unbundling"] = self.agents["unbundling"].analyze(claim_data)
            results["identity"] = self.agents["identity"].analyze(claim_data, id_path)
        except Exception as e:
            print(f"Error in sequential execution: {e}")
        return results

    def _run_mixed(self, claim_data: Dict, historical_data: Optional[List[Dict]], id_path: Optional[str]) -> Dict:
        """Run critical agents (identity) sequentially, others in parallel."""
        results = {}
        # Critical: Identity check first
        try:
            results["identity"] = self.agents["identity"].analyze(claim_data, id_path)
        except Exception as e:
            print(f"Error in identity agent: {e}")
            results["identity"] = None

        # Parallel: Other agents
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.agents["overbilling"].analyze, claim_data): "overbilling",
                executor.submit(self.agents["diagnostic"].analyze, claim_data, historical_data): "diagnostic",
                executor.submit(self.agents["unbundling"].analyze, claim_data): "unbundling",
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    print(f"Error in {agent_name} agent: {e}")
                    results[agent_name] = None
        return results

    def _compute_unified_score(self, agent_results: Dict, claim_data: Dict) -> Dict:
        """Compute unified fraud risk score from all agent results."""
        if not agent_results:
            return {"error": "No agent results"}

        # Aggregate findings
        all_findings = []
        all_suspicious = []
        risk_scores = []
        confidence_scores = []

        for agent_name, result in agent_results.items():
            if result is None:
                continue
            all_findings.extend(result.findings)
            all_suspicious.append({
                "agent": agent_name,
                "risk": result.risk_level.name,
                "confidence": result.confidence,
                "findings": result.findings
            })
            risk_scores.append(result.risk_level.value)
            confidence_scores.append(result.confidence)

        # Compute weighted fraud risk score (0-100)
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 1
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        fraud_score = int((avg_risk / 4.0) * 100 * (0.5 + avg_confidence * 0.5))

        # Overall risk determination
        if fraud_score >= 75:
            overall_risk = "CRITICAL"
            action = "REJECT - Hold claim for investigation"
        elif fraud_score >= 50:
            overall_risk = "HIGH"
            action = "MANUAL_REVIEW - Detailed investigation required"
        elif fraud_score >= 25:
            overall_risk = "MEDIUM"
            action = "FLAG - Additional verification recommended"
        else:
            overall_risk = "LOW"
            action = "APPROVE - Routine processing"

        # Convert agent results to JSON-serializable format
        agent_results_serializable = {}
        for agent_name, result in agent_results.items():
            if result:
                result_dict = asdict(result)
                result_dict['risk_level'] = result.risk_level.name  # Convert Enum to string
                agent_results_serializable[agent_name] = result_dict
            else:
                agent_results_serializable[agent_name] = None

        return {
            "claim_data": claim_data,
            "fraud_risk_score": fraud_score,
            "overall_risk_level": overall_risk,
            "recommended_action": action,
            "agent_results": agent_results_serializable,
            "all_findings": all_findings,
            "suspicious_items": all_suspicious,
            "summary": {
                "total_findings": len(all_findings),
                "agents_flagged": len([r for r in agent_results.values() if r and r.risk_level.value >= 2]),
                "average_confidence": avg_confidence,
            }
        }
