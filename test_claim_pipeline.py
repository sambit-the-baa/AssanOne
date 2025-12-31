#!/usr/bin/env python3
"""
Test the enhanced OCR pipeline on actual claim PDFs.
Uses the existing extractor infrastructure with OCR enhancement.
"""

import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ocr_training"))

from agent.extractor import ocr_pdf


def process_claim_enhanced(pdf_path: str) -> dict:
    """
    Process a claim PDF using enhanced OCR pipeline.
    
    Steps:
    1. Extract raw text using existing Extractor (PyMuPDF + Tesseract fallback)
    2. Apply OCR enhancement (preprocessing, postprocessing)
    3. Extract structured fields using INSURANCE_PATTERNS
    
    Args:
        pdf_path: Path to the claim PDF
        
    Returns:
        Dictionary with extracted fields and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print('='*60)
    
    # Step 1: Extract raw text using ocr_pdf function
    print("\n[Step 1] Extracting raw text from PDF...")
    raw_text = ocr_pdf(pdf_path)
    print(f"  - Extracted {len(raw_text)} characters")
    
    if not raw_text:
        print("  ❌ No text extracted!")
        return {'error': 'No text extracted', 'pdf_path': pdf_path}
    
    # Step 2: Apply OCR enhancement and extract fields
    print("\n[Step 2] Applying OCR enhancement and extracting fields...")
    try:
        from ocr_training.ocr_integration import OCREnhancer
        
        enhancer = OCREnhancer()
        
        # Use enhance_extraction which combines postprocessing + extraction
        final_fields = enhancer.enhance_extraction(raw_text)
        enhanced_text = raw_text  # Enhancement is embedded in extraction
        
        print(f"  - ML available: {enhancer.ml_available}")
        print(f"  - Fields extracted: {len([v for v in final_fields.values() if v])}")
        
    except Exception as e:
        import traceback
        print(f"  ⚠ Enhancement error: {e}")
        traceback.print_exc()
        enhanced_text = raw_text
        final_fields = {}
    
    # Print extracted fields
    print("\n" + "-"*40)
    print("EXTRACTED FIELDS:")
    print("-"*40)
    for field, value in final_fields.items():
        if value:
            print(f"  {field}: {value}")
    
    # Build result
    result = {
        'pdf_path': str(pdf_path),
        'raw_text_length': len(raw_text),
        'enhanced_text_length': len(enhanced_text),
        'extraction_method': 'pymupdf',
        'fields': final_fields,
    }
    
    # Sample of raw text for debugging
    print("\n" + "-"*40)
    print("SAMPLE TEXT (first 1000 chars):")
    print("-"*40)
    print(enhanced_text[:1000])
    
    return result


def main():
    """Process all claim PDFs in the Data folder."""
    data_dir = Path(__file__).parent / "Data"
    
    # Find all PDFs
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files in {data_dir}")
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    results = []
    for pdf_file in pdf_files:  # Process all PDFs
        result = process_claim_enhanced(str(pdf_file))
        results.append(result)
        
        # Save individual result
        output_path = data_dir / "outputs" / f"{pdf_file.stem}_enhanced.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {Path(r['pdf_path']).name}:")
        print(f"    - Text: {r['raw_text_length']} chars")
        print(f"    - Fields extracted: {len([v for v in r.get('fields', {}).values() if v])}")


if __name__ == "__main__":
    main()
