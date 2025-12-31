import json
import os
from pathlib import Path
from typing import Dict, Any

from agent.extractor import ocr_pdf
from agent.extractor_enhanced import EnhancedClaimExtractor


def ensure_outputs_dir(base: str = "Data/outputs") -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def process_pdf(path: str, run_both: bool = False, dpi: int = 300) -> Dict[str, Any]:
    """
    Process a PDF and save extracted fields as JSON in `Data/outputs`.

    If `run_both` is True, the function will run both OCR methods (pytesseract
    and Google Vision where available) and save separate JSON files with
    suffixes `_tesseract.json` and `_vision.json`.
    Returns a dictionary with keys for each saved result and their file paths.
    """
    outputs = ensure_outputs_dir()
    src = Path(path)
    stem = src.stem

    results: Dict[str, Any] = {}


    # Use enhanced extractor
    extractor = EnhancedClaimExtractor()
    text_auto = ocr_pdf(path, use_vision_if_available=True, dpi=dpi)
    fields_auto = extractor.extract_fields(text_auto)
    # Convert ExtractionResult objects to dicts for JSON serialization
    fields_auto_json = {k: (asdict(v) if hasattr(v, '__dataclass_fields__') else v) for k, v in fields_auto.items()}
    out_auto = outputs / f"{stem}.json"
    with open(out_auto, "w") as fh:
        json.dump(fields_auto_json, fh, indent=2, ensure_ascii=False)
    results["auto"] = {"fields": fields_auto_json, "path": str(out_auto)}

    if run_both:
        # Explicit pytesseract run
        text_tess = ocr_pdf(path, use_vision_if_available=False, dpi=dpi)
        fields_tess = extractor.extract_fields(text_tess)
        fields_tess_json = {k: (asdict(v) if hasattr(v, '__dataclass_fields__') else v) for k, v in fields_tess.items()}
        out_t = outputs / f"{stem}_tesseract.json"
        with open(out_t, "w") as fh:
            json.dump(fields_tess_json, fh, indent=2, ensure_ascii=False)
        results["tesseract"] = {"fields": fields_tess_json, "path": str(out_t)}

        # Explicit vision run if credentials available
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            text_vis = ocr_pdf(path, use_vision_if_available=True, dpi=dpi)
            fields_vis = extractor.extract_fields(text_vis)
            fields_vis_json = {k: (asdict(v) if hasattr(v, '__dataclass_fields__') else v) for k, v in fields_vis.items()}
            out_v = outputs / f"{stem}_vision.json"
            with open(out_v, "w") as fh:
                json.dump(fields_vis_json, fh, indent=2, ensure_ascii=False)
            results["vision"] = {"fields": fields_vis_json, "path": str(out_v)}

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agent.processor <pdf-path> [--both]")
    else:
        p = sys.argv[1]
        run_both = "--both" in sys.argv
        r = process_pdf(p, run_both=run_both)
        print(json.dumps(r, indent=2))
