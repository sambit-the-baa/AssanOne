# TPA Claims OCR Agent

This project provides a simple OCR-based agent to extract key fields from insurance claim PDF documents and present them on a small dashboard for TPA review.

Features
- PDF -> image conversion (pdf2image)
- OCR using Google Cloud Vision (if credentials are configured) or local `pytesseract` fallback
- Heuristic field extraction (policy number, claim number, claimant name, DOB, date of loss, amounts, provider, diagnosis)
- Streamlit dashboard to upload PDFs, review/edit extracted fields, and download results

Quick start (Windows)

1. Ensure system dependencies are installed:

- Tesseract OCR (for local OCR fallback):

  - Install from https://github.com/tesseract-ocr/tesseract or via Chocolatey:

```powershell
choco install tesseract
```

- Poppler (for `pdf2image` to convert PDFs to images):

  - Download poppler for Windows and add `bin` to your PATH, or install via Chocolatey:

```powershell
choco install poppler
```

2. Activate the virtual environment (created earlier):

```powershell
& "C:/One Intelligcnc agents/.venv/Scripts/Activate.ps1"
```

3. Install Python dependencies (if you haven't already):

```powershell
python -m pip install -r requirements-google-agents.txt
```

4. Optional: Configure Google Vision credentials to use Cloud OCR (better accuracy for complex documents):

- Create a service account and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to the JSON key file.

5. Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

Notes & next steps
- Extraction is heuristic-based. For higher accuracy, consider training an information-extraction model (layout-aware models such as LayoutLMv3, or using Google Document AI) and integrating it into `agent/extractor.py`.
- If you want I can add: a) CSV/DB export, b) bulk processing for entire `Data/` folder, or c) integration with Google Document AI (if you provide project credentials).

Auto-save and repeated uploads
- The app saves uploaded PDFs to `Data/uploads/` and writes extracted JSON results to `Data/outputs/`.
- Use the "Run both OCR methods" checkbox to save both `*_tesseract.json` and `*_vision.json` when available.
- Use the "Process all PDFs in Data folder" button to bulk-process existing PDFs in `Data/`.

Automation notes
- The processor will choose Google Vision when `GOOGLE_APPLICATION_CREDENTIALS` is set and the `google.cloud.vision` client is available. Otherwise it falls back to `pytesseract`.
- For best results, install system-level `tesseract` and `poppler` (see earlier), or configure Google credentials to use Cloud OCR.