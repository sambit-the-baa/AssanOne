"""
Test enhanced OCR pipeline on actual claim PDFs.
Uses the existing extractor infrastructure with the new preprocessing/postprocessing.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from agent.extractor import ocr_pdf, extract_fields

# Import enhanced components
try:
    from ocr_integration import OCREnhancer
    ENHANCER_AVAILABLE = True
    logger.info("OCR Enhancer module loaded")
except ImportError as e:
    ENHANCER_AVAILABLE = False
    logger.warning(f"OCR Enhancer not available: {e}")


def process_claim_enhanced(pdf_path: str) -> dict:
    """
    Process a claim PDF with enhanced OCR pipeline.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    logger.info(f"Processing: {pdf_path.name}")
    
    # Step 1: Extract raw text using existing OCR methods
    logger.info("Step 1: Extracting raw text...")
    raw_text = ocr_pdf(str(pdf_path))
    logger.info(f"  Raw text length: {len(raw_text)} chars")
    
    # Step 2: Apply enhancement if available
    if ENHANCER_AVAILABLE:
        logger.info("Step 2: Applying OCR enhancement...")
        enhancer = OCREnhancer()
        enhanced_result = enhancer.enhance_extraction(raw_text)
        enhanced_text = enhanced_result.get('processed_text', raw_text)
        ml_fields = enhanced_result.get('fields', {})
        logger.info(f"  Enhanced text length: {len(enhanced_text)} chars")
        logger.info(f"  ML extracted fields: {list(ml_fields.keys())}")
    else:
        enhanced_text = raw_text
        ml_fields = {}
    
    # Step 3: Extract fields using patterns
    logger.info("Step 3: Extracting fields with patterns...")
    pattern_fields = extract_fields(enhanced_text)
    
    # Step 4: Merge results (ML fields take precedence if high confidence)
    final_fields = pattern_fields.copy()
    for field, data in ml_fields.items():
        if isinstance(data, dict):
            value = data.get('value', '')
            confidence = data.get('confidence', 0)
            # Use ML result if confidence is high or pattern didn't find it
            if confidence > 0.6 or not final_fields.get(field):
                final_fields[field] = value
        elif data:
            if not final_fields.get(field):
                final_fields[field] = str(data)
    
    return {
        'file': pdf_path.name,
        'claim_id': pdf_path.stem,
        'raw_text_length': len(raw_text),
        'enhanced_text_length': len(enhanced_text),
        'enhancement_applied': ENHANCER_AVAILABLE,
        'fields': final_fields,
        'ml_fields': ml_fields,
        'raw_text_preview': raw_text[:500] + '...' if len(raw_text) > 500 else raw_text
    }


def main():
    """Process all claim PDFs in Data folder"""
    data_dir = project_root / 'Data'
    output_dir = data_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDFs
    pdfs = list(data_dir.glob('*.pdf'))
    logger.info(f"Found {len(pdfs)} PDF files")
    
    results = []
    for pdf_path in pdfs[:1]:  # Process first PDF as test
        try:
            result = process_claim_enhanced(str(pdf_path))
            results.append(result)
            
            # Print summary
            print("\n" + "=" * 60)
            print(f"CLAIM: {result['claim_id']}")
            print("=" * 60)
            print(f"Enhancement Applied: {result['enhancement_applied']}")
            print(f"Text: {result['raw_text_length']} → {result['enhanced_text_length']} chars")
            
            print("\nExtracted Fields:")
            fields = result['fields']
            priority_fields = [
                'claimant_name', 'patient_name', 'policy_number', 
                'provider_name', 'hospital_name', 'diagnosis',
                'date_of_admission', 'date_of_discharge', 'total_amount'
            ]
            for field in priority_fields:
                value = fields.get(field, 'NOT FOUND')
                if value:
                    print(f"  {field}: {value}")
            
            # Save result
            output_file = output_dir / f"{result['claim_id']}_enhanced_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                # Remove raw_text_preview for cleaner output
                save_data = {k: v for k, v in result.items() if k != 'raw_text_preview'}
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    main()
