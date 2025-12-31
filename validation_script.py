import json
from pathlib import Path

from agent.agents_fraud import (
    cross_verify_identity,
    check_hospital_vs_pharmacy,
    check_trimalleolar_orif,
    IdentityTheftProtectionAgent,
    OverbillingProtectionAgent,
    FraudDiagnosticAgent,
    UnbundlingUpccodingAgent,
    ClinicalConsistencyAgent,
    FinancialReconciliationAgent,
    compute_explainable_score,
)


DATA_DIR = Path("Data") / "outputs"
CLAIM_FILE = DATA_DIR / "1398305_claim.json"
FRAUD_FILE = DATA_DIR / "1398305_fraud_report.json"


def load_json(p: Path):
    if not p.exists():
        print(f"File not found: {p}")
        return None
    with p.open("r") as f:
        return json.load(f)


def main():
    claim = load_json(CLAIM_FILE)
    fraud = load_json(FRAUD_FILE)

    if claim is None:
        print("Claim JSON missing. Aborting validation.")
        return

    print("\n--- Cross-Document Identity Verification ---")
    cdv = cross_verify_identity(claim)
    print("Agreement:", cdv.get("agreement"))
    print("Resolved name:", cdv.get("resolved_name"))
    print("Resolved dob:", cdv.get("resolved_dob"))
    print("Name mismatch:", cdv.get("name_mismatch"))
    print("DOB mismatch:", cdv.get("dob_mismatch"))
    print("Sources:")
    for k, v in (cdv.get("sources") or {}).items():
        print(f"  {k}: {v}")

    print("\n--- Hospital vs Pharmacy Check ---")
    hb = check_hospital_vs_pharmacy(claim)
    print(json.dumps(hb, indent=2))

    print("\n--- Diagnostic Check (Trimalleolar / ORIF) ---")
    tr = check_trimalleolar_orif(claim)
    print(json.dumps(tr, indent=2))

    print("\n--- Quick Assertions for Kate Claim ---")
    # Patient identity
    resolved_name = cdv.get("resolved_name")
    resolved_dob = cdv.get("resolved_dob")
    if resolved_name and resolved_dob:
        print("PASS: Patient name and DOB resolved from documents.")
    else:
        print("FAIL: Patient name or DOB NOT resolved across documents.")

    # Run enhanced agents locally for explainable checks
    agents = []
    agents.append(IdentityTheftProtectionAgent())
    agents.append(OverbillingProtectionAgent())
    agents.append(FraudDiagnosticAgent())
    agents.append(UnbundlingUpccodingAgent())
    agents.append(ClinicalConsistencyAgent())
    agents.append(FinancialReconciliationAgent())

    results = []
    for a in agents:
        try:
            # some analyze methods accept different params
            if a.__class__.__name__ == 'IdentityTheftProtectionAgent':
                res = a.analyze(claim)
            elif a.__class__.__name__ == 'FraudDiagnosticAgent':
                res = a.analyze(claim, historical_data=None)
            else:
                res = a.analyze(claim)
        except Exception as e:
            print(f"Error running agent {a.__class__.__name__}: {e}")
            continue
        # convert dataclass to dict-like for printing
        out = {
            "agent_name": res.agent_name,
            "risk_level": res.risk_level.name,
            "confidence": res.confidence,
            "findings": res.findings,
            "details": res.details,
            "recommended_action": res.recommended_action,
        }
        print(f"\nAgent result: {res.agent_name}")
        print(json.dumps(out, indent=2, default=str))
        results.append(res)

    # Compute explainable aggregated score
    expl = compute_explainable_score(results)
    print('\n--- Explainable Aggregated Score ---')
    print(json.dumps(expl, indent=2, default=str))

    print("\nValidation complete.")


if __name__ == '__main__':
    main()
