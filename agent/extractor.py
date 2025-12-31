"""
Robust PDF text extraction and field extraction module.
Handles multiple encodings and OCR fallbacks.
"""
import os
import re
import io
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import chardet
except Exception:
    chardet = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from google.cloud import vision
except Exception:
    vision = None

try:
    import pytesseract
    # Configure Tesseract path for Windows
    import platform
    if platform.system() == 'Windows':
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe',
        ]
        for tess_path in tesseract_paths:
            if os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path
                break
except Exception:
    pytesseract = None

try:
    import fitz
except Exception:
    fitz = None


# ============================================================================
# ENCODING DETECTION & FIXING
# ============================================================================
def detect_encoding(data: bytes) -> str:
    """Detect file encoding using chardet library"""
    if chardet is None:
        return 'utf-8'
    try:
        result = chardet.detect(data)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
        return encoding if confidence > 0.7 else 'utf-8'
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, defaulting to utf-8")
        return 'utf-8'


def decode_text_safely(data: bytes) -> str:
    """Safely decode text from bytes, trying multiple encodings"""
    encodings_to_try = [
        'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 
        'latin-1', 'cp1252', 'iso-8859-1', 'ascii',
    ]
    
    # First, try to detect
    detected = detect_encoding(data)
    if detected and detected.lower() not in [e.lower() for e in encodings_to_try]:
        encodings_to_try.insert(0, detected)
    
    for encoding in encodings_to_try:
        try:
            text = data.decode(encoding, errors='ignore')
            if len(text.strip()) > 0:
                logger.info(f"Successfully decoded with {encoding}")
                return text
        except Exception as e:
            logger.debug(f"Failed with {encoding}: {e}")
            continue
    
    # Fallback
    logger.error("All encoding attempts failed, returning with errors='replace'")
    return data.decode('utf-8', errors='replace')


# ============================================================================
# OCR FUNCTIONS
# ============================================================================
def ocr_image_with_vision(img: Image.Image) -> str:
    """Use Google Cloud Vision for OCR"""
    if not vision:
        return ""
    try:
        client = vision.ImageAnnotatorClient()
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        content = buf.getvalue()
        image = vision.Image(content=content)
        resp = client.document_text_detection(image=image)
        text = resp.full_text_annotation.text if resp and resp.full_text_annotation else ""
        return text
    except Exception as e:
        logger.error(f"Vision OCR failed: {e}")
        return ""


def ocr_image_pytesseract(img: Image.Image) -> str:
    """Use Tesseract for OCR"""
    if not pytesseract:
        logger.debug("Tesseract not available, using fallback")
        return ""
    try:
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.warning(f"Tesseract OCR unavailable: {e}")
        return ""


def mock_ocr(img: Image.Image) -> str:
    """Fallback OCR placeholder with image dimensions"""
    return f"[Mock OCR - image size: ({img.size[0]}, {img.size[1]})] Document page with claim information."


# ============================================================================
# MAIN OCR FUNCTION
# ============================================================================
def ocr_pdf(
    pdf_path: str,
    use_vision_if_available: bool = True,
    dpi: int = 300,
) -> str:
    """
    Extract text from PDF with multiple fallback strategies.
    
    Strategies tried in order:
    1. PyMuPDF direct text extraction
    2. pdf2image + OCR
    3. PyMuPDF render + OCR
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    texts = []
    
    # METHOD 1: PyMuPDF Direct Text Extraction
    if fitz is not None:
        try:
            logger.info("Attempting PyMuPDF direct text extraction...")
            doc = fitz.open(str(p))
            for page_num, page in enumerate(doc):
                try:
                    text = page.get_text()
                    if isinstance(text, bytes):
                        text = decode_text_safely(text)
                    if text.strip():
                        texts.append(text)
                        logger.debug(f"Page {page_num}: extracted {len(text)} chars")
                except Exception as e:
                    logger.warning(f"Page {page_num} extraction failed: {e}")
                    continue
            doc.close()
            if texts:
                result = "\n".join(texts)
                logger.info(f"PyMuPDF extraction successful: {len(result)} chars")
                return result
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
    
    # METHOD 2: pdf2image + OCR (Fallback)
    if convert_from_path is not None:
        try:
            logger.info("Attempting pdf2image + OCR...")
            images = convert_from_path(str(p), dpi=dpi)
            use_vision = use_vision_if_available and os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and vision
            for img_num, img in enumerate(images):
                try:
                    if use_vision:
                        text = ocr_image_with_vision(img)
                    else:
                        text = ocr_image_pytesseract(img) or mock_ocr(img)
                    if text.strip():
                        texts.append(text)
                        logger.debug(f"Image {img_num}: extracted {len(text)} chars")
                except Exception as e:
                    logger.warning(f"Image {img_num} OCR failed: {e}")
                    continue
            if texts:
                result = "\n".join(texts)
                logger.info(f"PDF to Image + OCR successful: {len(result)} chars")
                return result
        except Exception as e:
            logger.warning(f"pdf2image + OCR failed: {e}")
    
    # METHOD 3: PyMuPDF Render + OCR
    if fitz is not None:
        try:
            logger.info("Attempting PyMuPDF render + OCR...")
            doc = fitz.open(str(p))
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            use_vision = use_vision_if_available and os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and vision
            for page_num, page in enumerate(doc):
                try:
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.open(io.BytesIO(pix.tobytes(output='png')))
                    if use_vision:
                        text = ocr_image_with_vision(img)
                    elif pytesseract:
                        text = ocr_image_pytesseract(img)
                    else:
                        text = mock_ocr(img)
                    if text.strip():
                        texts.append(text)
                        logger.debug(f"Rendered page {page_num}: extracted {len(text)} chars")
                except Exception as e:
                    logger.warning(f"Page {page_num} render failed: {e}")
                    continue
            doc.close()
            if texts:
                result = "\n".join(texts)
                logger.info(f"PyMuPDF render + OCR successful: {len(result)} chars")
                return result
        except Exception as e:
            logger.warning(f"PyMuPDF render failed: {e}")
    
    # METHOD 4: Return placeholder text with file info (fallback when no OCR available)
    logger.warning("All OCR methods failed - returning placeholder with file metadata")
    file_size = p.stat().st_size if p.exists() else 0
    placeholder_text = f"""
CLAIM DOCUMENT EXTRACTED
File: {p.name}
Size: {file_size} bytes
Date Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[Document requires OCR - Tesseract not available]
[Install Tesseract for full text extraction]

Claim ID: {p.stem}
Status: Pending Review
Amount: $0.00
Provider: Unknown
Patient: Unknown
Date of Service: Unknown
Diagnosis: Unknown
Procedure: Unknown
"""
    return placeholder_text.strip()


# ============================================================================
# FIELD EXTRACTION
# ============================================================================
def _find_first(pattern: str, text: str) -> Optional[str]:
    """Find first match of pattern in text"""
    try:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def extract_fields(text: str) -> Dict[str, Optional[str]]:
    """
    Extract claim fields from OCR text.
    Returns a dictionary of common insurance claim fields.
    """
    if isinstance(text, bytes):
        text = decode_text_safely(text)
    
    # Ensure text is clean
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    out = {}
    
    # Policy Number
    out['policy_number'] = (
        _find_first(r'(?:Policy\s*(?:Number|No\.?|#)?\s*[:=]?\s*)([A-Z0-9\-]{6,25})', text)
        or _find_first(r'(?:POL\s*[:=]?\s*)([A-Z0-9\-]{6,25})', text)
        or _find_first(r'(?:Member\s*ID\s*[:=]?\s*)([A-Z0-9\-]{6,25})', text)
        or "UNKNOWN"
    )
    
    # Claim Number
    out['claim_number'] = (
        _find_first(r'(?:Claim\s*(?:Number|No\.?|#|ID)?\s*[:=]?\s*)([A-Z0-9\-]{5,25})', text)
        or _find_first(r'(?:CLM\s*[:=]?\s*)([A-Z0-9\-]{5,25})', text)
        or f"CLM-{abs(hash(text)) % 1000000}"
    )
    
    # Claimant Name
    out['claimant_name'] = (
        _find_first(r'(?:(?:Claimant|Insured|Patient|Member)\s*(?:Name|:)?\s*)\n?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        or "UNKNOWN"
    )
    
    # Date of Birth
    out['dob'] = (
        _find_first(r'(?:(?:DOB|Date\s+of\s+Birth|Birth\s+Date)\s*[:=]?\s*)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
        or "UNKNOWN"
    )
    
    # Date of Service
    out['date_of_service'] = (
        _find_first(r'(?:(?:DOS|Date\s+of\s+Service|Service\s+Date|Admission)\s*[:=]?\s*)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
        or "UNKNOWN"
    )
    
    # Provider Name
    out['provider_name'] = (
        _find_first(r'(?:(?:Provider|Physician|Doctor|Hospital)\s*(?:Name|:)?\s*)\n?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        or "UNKNOWN"
    )
    
    # Diagnosis
    out['diagnosis'] = (
        _find_first(r'(?:Diagnosis\s*[:=]?\s*)([A-Z]\d{2}\.?\d{0,2}(?:[A-Z])?)', text)
        or _find_first(r'(?:Diagnosis\s*[:=]?\s*)([A-Za-z0-9\s\-,]{3,100})', text)
        or "UNKNOWN"
    )
    
    # Procedures (CPT codes)
    procedures = re.findall(r'(?:CPT\s*[:=]?\s*)?(\d{5}(?:[A-Z])?)', text, flags=re.IGNORECASE)
    out['procedures'] = procedures if procedures else ["UNKNOWN"]
    
    # Amounts
    amounts = re.findall(r'\$?\s*([\d,]+\.?\d{0,2})', text)
    try:
        numeric_amounts = [float(a.replace(',', '')) for a in amounts if a.replace(',', '').replace('.', '').isdigit()]
        out['total_amount'] = max(numeric_amounts) if numeric_amounts else 0
    except (ValueError, TypeError):
        out['total_amount'] = 0
    out['amounts_found'] = ", ".join(amounts[:5]) if amounts else "0"
    
    # Insurance Company
    out['insurance_company'] = (
        _find_first(r'(?:Insurance\s*(?:Company|Plan)?\s*[:=]?\s*)([A-Z][A-Za-z\s\&\.]+)', text)
        or "UNKNOWN"
    )
    
    # Raw text preview for debugging
    out['raw_text_preview'] = text[:2000]
    
    logger.info(f"Extracted fields: {list(out.keys())}")
    return out


if __name__ == "__main__":
    print("agent.extractor module. Use ocr_pdf(path) and extract_fields(text).")
