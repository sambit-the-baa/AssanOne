"""
Enhanced OCR Pipeline
Integrates all components:
1. Image preprocessing (deskew, denoise, threshold)
2. Form layout detection (boxes, lines, checkboxes)
3. Field-specific extraction (printed, handwritten, checkbox)
4. Text postprocessing (normalization, typo correction)
5. BERT NER for entity extraction
"""

import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import json

# Import pipeline components
from preprocessing import ImagePreprocessor, PreprocessingConfig
from layout_detection import FormLayoutDetector, FormLayout, FieldType
from field_extractors import FieldExtractorPipeline, PrintedTextExtractor
from text_postprocessing import TextPostprocessor, PostprocessingConfig
from bert_ner_inference import HybridExtractor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the enhanced OCR pipeline"""
    # Preprocessing
    dpi: int = 350
    blur_kernel: tuple = (3, 3)
    adaptive_block_size: int = 31
    
    # Layout detection
    detect_checkboxes: bool = True
    detect_tables: bool = True
    classify_text_type: bool = True
    
    # Field extraction
    use_handwriting_model: bool = False  # TrOCR requires transformers
    
    # Postprocessing
    fix_ocr_patterns: bool = True
    domain_corrections: bool = True
    
    # NER
    ner_model_path: str = "ocr_training/models/bert_ner"
    ner_confidence_threshold: float = 0.6
    
    # Output
    save_intermediate: bool = False
    output_dir: str = "ocr_output"


@dataclass
class ExtractionResult:
    """Result of OCR extraction"""
    raw_text: str
    processed_text: str
    fields: Dict[str, Any]
    layout: Optional[FormLayout] = None
    checkboxes: Dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedOCRPipeline:
    """
    Complete OCR pipeline for insurance claim forms.
    Handles scanned PDFs/images with checkboxes, printed and handwritten text.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components"""
        # Preprocessing
        preprocess_config = PreprocessingConfig(
            dpi=self.config.dpi,
            blur_kernel=self.config.blur_kernel,
            adaptive_block_size=self.config.adaptive_block_size
        )
        self.preprocessor = ImagePreprocessor(preprocess_config)
        
        # Layout detection
        self.layout_detector = FormLayoutDetector()
        
        # Field extraction
        self.field_extractor = FieldExtractorPipeline(
            use_handwriting_model=self.config.use_handwriting_model
        )
        
        # Postprocessing
        postprocess_config = PostprocessingConfig(
            fix_ocr_patterns=self.config.fix_ocr_patterns,
            domain_corrections=self.config.domain_corrections
        )
        self.postprocessor = TextPostprocessor(postprocess_config)
        
        # NER extractor
        ner_path = self.config.ner_model_path
        if os.path.exists(ner_path):
            self.ner_extractor = HybridExtractor(
                ner_model_path=ner_path,
                confidence_threshold=self.config.ner_confidence_threshold
            )
        else:
            logger.warning(f"NER model not found at {ner_path}, using regex-only extraction")
            self.ner_extractor = HybridExtractor(confidence_threshold=0.5)
        
        # Printed text extractor for full-page OCR
        self.printed_extractor = PrintedTextExtractor()
        
        logger.info("OCR Pipeline initialized")
    
    def process_image(self, image: np.ndarray) -> ExtractionResult:
        """
        Process a single image through the full pipeline.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            ExtractionResult with extracted data
        """
        metadata = {'image_shape': image.shape}
        
        # Step 1: Preprocess image
        logger.info("Step 1: Preprocessing image...")
        preprocessed = self.preprocessor.preprocess_full(image)
        binary_image = preprocessed['binary']
        metadata['preprocessing'] = {
            'deskew_angle': preprocessed.get('deskew_angle', 0),
            'noise_removed': preprocessed.get('noise_removed', False)
        }
        
        # Step 2: Detect form layout
        logger.info("Step 2: Detecting form layout...")
        layout = self.layout_detector.analyze_form(binary_image)
        metadata['layout'] = {
            'num_fields': len(layout.fields),
            'num_checkboxes': sum(1 for f in layout.fields if f.field_type == FieldType.CHECKBOX),
            'has_tables': layout.has_tables
        }
        
        # Step 3: Extract field contents
        logger.info("Step 3: Extracting field contents...")
        self.field_extractor.extract_all_fields(layout.fields, binary_image)
        
        # Collect checkbox states
        checkboxes = {}
        for f in layout.fields:
            if f.field_type == FieldType.CHECKBOX:
                checkboxes[f.label or f"checkbox_{f.bbox.x}_{f.bbox.y}"] = f.extracted_text == "checked"
        
        # Step 4: Full-page OCR for text extraction
        logger.info("Step 4: Running full-page OCR...")
        full_page_result = self.printed_extractor.extract_with_multiple_configs(binary_image)
        raw_text = full_page_result['text']
        
        # Step 5: Postprocess text
        logger.info("Step 5: Postprocessing text...")
        postprocessed = self.postprocessor.postprocess(raw_text, 'general')
        processed_text = postprocessed['processed']
        
        # Step 6: Run NER for entity extraction
        logger.info("Step 6: Running NER extraction...")
        ner_result = self.ner_extractor.extract(processed_text)
        
        # Step 7: Apply field-specific postprocessing to NER results
        fields = {}
        for field_name, data in ner_result.items():
            # Apply field-specific corrections
            field_result = self.postprocessor.postprocess(data['value'], field_name)
            fields[field_name] = {
                'value': field_result['processed'],
                'raw_value': data['value'],
                'source': data['source'],
                'confidence': data['confidence']
            }
        
        # Calculate overall confidence
        if fields:
            avg_confidence = np.mean([f['confidence'] for f in fields.values()])
        else:
            avg_confidence = full_page_result.get('confidence', 0)
        
        return ExtractionResult(
            raw_text=raw_text,
            processed_text=processed_text,
            fields=fields,
            layout=layout,
            checkboxes=checkboxes,
            confidence=avg_confidence,
            metadata=metadata
        )
    
    def process_pdf(self, pdf_path: str) -> List[ExtractionResult]:
        """
        Process a PDF document.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of ExtractionResult for each page
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = self.preprocessor.pdf_to_images(pdf_path)
        logger.info(f"Converted PDF to {len(images)} images")
        
        # Process each page
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            result = self.process_image(image)
            result.metadata['page_number'] = i + 1
            results.append(result)
        
        return results
    
    def process_pdf_merged(self, pdf_path: str) -> ExtractionResult:
        """
        Process PDF and merge all pages into single result.
        Useful for multi-page claim forms.
        """
        page_results = self.process_pdf(pdf_path)
        
        if len(page_results) == 1:
            return page_results[0]
        
        # Merge results
        merged_raw = "\n\n--- PAGE BREAK ---\n\n".join(r.raw_text for r in page_results)
        merged_processed = "\n\n".join(r.processed_text for r in page_results)
        
        # Merge fields (take highest confidence for duplicates)
        merged_fields = {}
        for result in page_results:
            for field_name, data in result.fields.items():
                if field_name not in merged_fields or data['confidence'] > merged_fields[field_name]['confidence']:
                    merged_fields[field_name] = data
        
        # Merge checkboxes
        merged_checkboxes = {}
        for result in page_results:
            merged_checkboxes.update(result.checkboxes)
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in page_results])
        
        return ExtractionResult(
            raw_text=merged_raw,
            processed_text=merged_processed,
            fields=merged_fields,
            checkboxes=merged_checkboxes,
            confidence=avg_confidence,
            metadata={'num_pages': len(page_results), 'page_results': page_results}
        )
    
    def extract_claim_data(self, source: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Main entry point for claim extraction.
        Accepts PDF path or image array.
        
        Returns structured claim data ready for downstream processing.
        """
        # Process source
        if isinstance(source, str):
            if source.lower().endswith('.pdf'):
                result = self.process_pdf_merged(source)
            else:
                # Assume image file
                image = cv2.imread(source)
                if image is None:
                    raise ValueError(f"Failed to load image: {source}")
                result = self.process_image(image)
        else:
            result = self.process_image(source)
        
        # Convert to claim data format
        claim_data = {
            'extraction_status': 'success',
            'confidence': result.confidence,
            'fields': {}
        }
        
        # Map to standard claim fields
        field_mapping = {
            'claimant_name': 'patient_name',
            'policy_number': 'policy_number',
            'hospital_name': 'hospital_name',
            'admission_date': 'admission_date',
            'discharge_date': 'discharge_date',
            'diagnosis': 'diagnosis',
            'amount': 'claim_amount',
            'claim_id': 'claim_id',
            'doctor_name': 'treating_doctor',
            'address': 'hospital_address',
        }
        
        for source_field, target_field in field_mapping.items():
            if source_field in result.fields:
                claim_data['fields'][target_field] = result.fields[source_field]['value']
        
        # Add checkboxes
        claim_data['checkboxes'] = result.checkboxes
        
        # Add full text for reference
        claim_data['full_text'] = result.processed_text
        
        return claim_data
    
    def save_result(self, result: ExtractionResult, output_path: str):
        """Save extraction result to JSON"""
        output = {
            'raw_text': result.raw_text,
            'processed_text': result.processed_text,
            'fields': result.fields,
            'checkboxes': result.checkboxes,
            'confidence': result.confidence,
            'metadata': {k: v for k, v in result.metadata.items() 
                        if not isinstance(v, (np.ndarray, list))}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved result to {output_path}")


def create_pipeline(
    dpi: int = 350,
    use_handwriting: bool = False,
    ner_model: str = None
) -> EnhancedOCRPipeline:
    """
    Factory function to create pipeline with common configurations.
    """
    config = PipelineConfig(
        dpi=dpi,
        use_handwriting_model=use_handwriting,
        ner_model_path=ner_model or "ocr_training/models/bert_ner"
    )
    return EnhancedOCRPipeline(config)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Enhanced OCR Pipeline")
    parser.add_argument("input", help="PDF or image file to process")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--dpi", type=int, default=350, help="DPI for PDF conversion")
    parser.add_argument("--ner-model", help="Path to NER model")
    parser.add_argument("--use-handwriting", action="store_true", 
                        help="Enable TrOCR for handwriting")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_pipeline(
        dpi=args.dpi,
        use_handwriting=args.use_handwriting,
        ner_model=args.ner_model
    )
    
    # Process input
    print(f"\nProcessing: {args.input}")
    print("=" * 60)
    
    try:
        claim_data = pipeline.extract_claim_data(args.input)
        
        print("\n=== EXTRACTION RESULTS ===\n")
        print(f"Status: {claim_data['extraction_status']}")
        print(f"Confidence: {claim_data['confidence']:.2%}")
        
        print("\nExtracted Fields:")
        for field, value in claim_data['fields'].items():
            print(f"  {field}: {value}")
        
        if claim_data['checkboxes']:
            print("\nCheckboxes:")
            for name, checked in claim_data['checkboxes'].items():
                status = "☑" if checked else "☐"
                print(f"  {status} {name}")
        
        # Save output if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(claim_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
