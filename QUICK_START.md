# Quick Start Guide - Fraud Detection System

## 1-Minute Setup

```powershell
# Activate environment
& "C:/One Intelligcnc agents/.venv/Scripts/Activate.ps1"

# Run dashboard
streamlit run fraud_detection_dashboard.py
```

Open browser to: `http://localhost:8501`

## 2-Minute Test

```powershell
# Activate environment
& "C:/One Intelligcnc agents/.venv/Scripts/Activate.ps1"

# Process sample claim (36-page PDF)
python -m agent.pipeline Data/1398305.pdf --mode parallel
```

**Output**: 
- Fraud Risk Score: **17/100** (LOW)
- Recommended Action: **APPROVE**
- Results saved to: `Data/outputs/1398305_fraud_report.json`

## System Overview

### What It Does

Analyzes insurance claims for fraud using 4 intelligent agents:

1. **Overbilling Protection** - Detects overcharging
2. **Fraud Diagnostic** - Finds diagnosis-procedure mismatches  
3. **Unbundling/Upcoding** - Catches separate billing of bundled services
4. **Identity Theft** - Verifies claimant identity & deepfakes

### How It Works

```
Upload PDF → OCR Extract → Run 4 Agents (Parallel) → Score → Action
```

**Fraud Risk Score**: 0-100
- 0-24: LOW (Approve)
- 25-49: MEDIUM (Flag for review)
- 50-74: HIGH (Manual review required)
- 75-100: CRITICAL (Hold & investigate)

### Execution Modes

| Mode | Speed | Best For |
|------|-------|----------|
| **Parallel** | ⚡⚡⚡ Fastest | Production/batch |
| **Sequential** | 🐢 Slowest | Debugging |
| **Mixed** | ⚡⚡ Balanced | Default |

## Usage Examples

### Via Dashboard (Easiest)

```powershell
streamlit run fraud_detection_dashboard.py
```

Then:
1. Upload PDF or select from Data folder
2. Click "Analyze Claim"
3. View results by agent
4. Download JSON report

### Via Command Line

```powershell
python -m agent.pipeline Data/1398305.pdf --mode parallel
```

### Via Python API

```python
from agent.pipeline import process_claim_full_pipeline

result = process_claim_full_pipeline(
    "Data/claim.pdf",
    execution_mode="parallel"
)

print(f"Risk Score: {result['fraud_risk_score']}/100")
print(f"Risk Level: {result['overall_risk_level']}")
```

## File Locations

**Input**: Place claim PDFs in `Data/` folder

**Output**: Results saved to `Data/outputs/`
- `*_claim.json` - Extracted claim fields
- `*_fraud_report.json` - Full fraud detection report

**Code**:
- `agent/extractor.py` - OCR & extraction
- `agent/agents_fraud.py` - 4 fraud agents
- `agent/orchestrator.py` - Parallel execution engine
- `agent/pipeline.py` - End-to-end processing

## Troubleshooting

### Dashboard won't start
```
Error: streamlit not found
→ Run: pip install streamlit
```

### PDF processing fails
```
Error: tesseract not installed
→ Run: choco install tesseract -y
→ Or: Will use mock OCR (fallback)
```

### Slow performance
```
Suggestion 1: Use parallel mode (faster)
Suggestion 2: Install Tesseract (20x faster than cloud)
Suggestion 3: Use sequential mode if parallel has issues
```

## Key Files Created

**Agents** (NEW):
- `agent/agents_fraud.py` - 4 fraud detection agents
- `agent/orchestrator.py` - Parallel/sequential execution
- `agent/pipeline.py` - End-to-end pipeline

**Dashboard** (NEW):
- `fraud_detection_dashboard.py` - Streamlit UI

**Documentation**:
- `FRAUD_DETECTION_SYSTEM.md` - Full architecture docs
- `QUICK_START.md` - This file

**Existing** (Enhanced):
- `agent/extractor.py` - Added fallback OCR strategies
- `requirements-google-agents.txt` - Added vision/ML libs

## What Can Be Extended

✓ **Real Data Sources**
- Government ID APIs (instead of mock)
- Medical pricing databases
- Claims deduplication database

✓ **Better ML Models**
- Deepfake detection (instead of mock)
- Layout-aware OCR (LayoutLMv3)
- Diagnosis-procedure relationship learning

✓ **Scaling**
- Distribute agents across multiple machines (Ray)
- Deploy as microservices (FastAPI)
- Database storage for audit trail

## Next Steps

1. **Test the system**
   ```
   Place your PDFs in Data/
   Run the dashboard or CLI
   Review the fraud reports
   ```

2. **Customize risk thresholds**
   - Edit `agent/orchestrator.py`
   - Adjust fraud score weights

3. **Connect real data sources**
   - Update agent classes with real APIs
   - Remove mock implementations

4. **Deploy to production**
   - Set up FastAPI server
   - Add authentication & audit logging
   - Connect to claims database

## Support

**Issues?**
- Check `FRAUD_DETECTION_SYSTEM.md` for detailed architecture
- Review agent code in `agent/agents_fraud.py`
- Check output JSON files in `Data/outputs/`

**Want to Add a New Agent?**
- See "Extending the System" in `FRAUD_DETECTION_SYSTEM.md`
- Add class to `agent/agents_fraud.py`
- Register in `agent/orchestrator.py`

---

**Status**: ✓ Ready to Use  
**Test Claim**: 1398305.pdf (36 pages)  
**Execution Modes**: Parallel ✓ Sequential ✓ Mixed ✓  
**Agents**: 4 (Overbilling, Diagnostic, Unbundling, Identity)
