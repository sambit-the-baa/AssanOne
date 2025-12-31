import os
import tempfile
import json
import shutil
from pathlib import Path

import streamlit as st


from agent.extractor import ocr_pdf
from agent.extractor_enhanced import EnhancedClaimExtractor
from agent.processor import process_pdf


st.set_page_config(page_title="Claims OCR Dashboard", layout="wide")

st.title("TPA Claims OCR — Extraction Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload / Choose PDF")
    uploaded = st.file_uploader("Upload claim PDF", type=["pdf"])

    data_dir = Path("Data")
    uploads_dir = data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    if data_dir.exists():
        st.write("Or pick from `Data` folder:")
        files = list(data_dir.glob("*.pdf"))
        for f in files:
            if st.button(f.name):
                uploaded = open(f, "rb")

    st.write("\n")
    ocr_choice = st.selectbox("OCR engine (auto chooses best available)", ["auto", "tesseract", "vision"])
    run_both = st.checkbox("Run both OCR methods (save both results if possible)")

    if st.button("Process uploaded PDF") and uploaded:
        pass

with col2:
    st.header("Extraction & Review")
    if uploaded:
        # ensure file saved
        if hasattr(uploaded, "read"):
            tmp = uploads_dir / (uploaded.name if hasattr(uploaded, "name") else ("upload.pdf"))
            with open(tmp, "wb") as fh:
                fh.write(uploaded.read())
            pdf_path = str(tmp)
        else:
            pdf_path = str(uploaded)

        st.info(f"Running OCR on: {Path(pdf_path).name}")
        with st.spinner("Performing OCR  this may take a moment..."):
            try:
                # If user requested both, let processor handle saving both outputs
                results = process_pdf(pdf_path, run_both=run_both)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                results = {}

        # Show results
        if results:
            st.subheader("Saved Results")
            for key, val in results.items():
                st.write(f"**{key}** — saved to: {val.get('path')}")
                st.download_button(f"Download {key} JSON", data=json.dumps(val.get("fields", {}), indent=2), file_name=Path(val.get("path")).name)

            # Show editable auto/default fields if present
            if "auto" in results:
                fields = results["auto"]["fields"]
                st.subheader("Extracted Fields (Auto)")
                edited = {}
                for k, v in fields.items():
                    if k == "raw_text_preview":
                        continue
                    edited[k] = st.text_input(k.replace("_", " ").title(), value=v or "")

                st.download_button("Download Edited JSON", data=json.dumps(edited, indent=2), file_name=f"{Path(pdf_path).stem}_edited.json")

            # Raw text preview from auto if available
            raw = results.get("auto", {}).get("fields", {}).get("raw_text_preview")
            if raw:
                st.subheader("Raw text preview")
                st.text_area("OCR text", value=raw, height=300)

    else:
        st.info("Upload a PDF to begin OCR and extraction, or pick from the `Data/` folder.")

    st.markdown("---")
    st.header("Bulk Processing")
    if data_dir.exists():
        if st.button("Process all PDFs in Data folder"):
            files = list(data_dir.glob("*.pdf"))
            if not files:
                st.warning("No PDFs found in Data folder.")
            else:
                with st.spinner(f"Processing {len(files)} files..."):
                    for f in files:
                        process_pdf(str(f), run_both=run_both)
                st.success("Bulk processing complete. Outputs saved to Data/outputs/")
