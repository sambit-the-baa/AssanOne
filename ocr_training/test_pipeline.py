"""
Test script for the enhanced OCR pipeline.
Tests all components: preprocessing, layout detection, field extraction,
postprocessing, and NER.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add ocr_training to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


def test_text_postprocessing():
    """Test text postprocessing module"""
    print("\n" + "=" * 60)
    print("Testing Text Postprocessing")
    print("=" * 60)
    
    from text_postprocessing import TextPostprocessor
    
    processor = TextPostprocessor()
    
    test_cases = [
        ("narne: SANDHYA RAGHUNATH KATE", "name"),
        ("Po1icy No: ABC12345678", "policy_number"),
        ("hospita1 treatrnent", "general"),
        ("Date: 15/O8/2024", "date"),
        ("Rs. 1,50,000.00", "amount"),
    ]
    
    for text, field_type in test_cases:
        result = processor.postprocess(text, field_type)
        print(f"\n  Input: '{text}'")
        print(f"  Type:  {field_type}")
        print(f"  Output: '{result['processed']}'")
        print(f"  Changed: {result['changed']}")
    
    return True


def test_ocr_noise_generator():
    """Test synthetic OCR noise generation"""
    print("\n" + "=" * 60)
    print("Testing OCR Noise Generator")
    print("=" * 60)
    
    from train_noisy_ner import OCRNoiseGenerator, OCRNoiseConfig
    
    config = OCRNoiseConfig(
        char_substitution_prob=0.1,
        word_split_prob=0.05
    )
    noise_gen = OCRNoiseGenerator(config)
    
    test_texts = [
        "Patient Name: Sandhya Kate",
        "Hospital: ACCIDENT HOSPITAL",
        "Policy No: ABC12345678",
    ]
    
    for text in test_texts:
        noisy = noise_gen.add_noise(text)
        print(f"\n  Original: '{text}'")
        print(f"  Noisy:    '{noisy}'")
    
    return True


def test_hybrid_extractor():
    """Test hybrid NER + regex extraction"""
    print("\n" + "=" * 60)
    print("Testing Hybrid Extractor")
    print("=" * 60)
    
    from bert_ner_inference import HybridExtractor
    
    # Create extractor (will use regex if model not found)
    model_path = script_dir / "models" / "bert_ner"
    extractor = HybridExtractor(
        str(model_path) if model_path.exists() else None,
        confidence_threshold=0.6
    )
    
    test_text = """
    Patient Name: SANDHYA RAGHUNATH KATE
    Policy No: ABC12345678
    Hospital: ACCIDENT HOSPITAL VADUJ
    Admission Date: 15/08/2024
    Discharge Date: 18/08/2024
    Diagnosis: Viral Fever
    Total Amount: Rs. 25,000.00
    """
    
    result = extractor.extract(test_text)
    
    print(f"\n  Input text length: {len(test_text)} chars")
    print("\n  Extracted fields:")
    for field, data in result.items():
        print(f"    {field}: {data['value']} (source: {data['source']}, conf: {data['confidence']:.2f})")
    
    return len(result) > 0


def test_ocr_enhancer():
    """Test full OCR enhancement integration"""
    print("\n" + "=" * 60)
    print("Testing OCR Enhancer (Full Integration)")
    print("=" * 60)
    
    from ocr_integration import OCREnhancer
    
    enhancer = OCREnhancer()
    
    # Simulate noisy OCR text
    noisy_text = """
    Pat1ent Narne: SANDHYA RAGHUNATH KATE
    Po1icy No: ABC12345678
    Hospita1: ACCIDENT HOSPITAL VADUJ
    Adrnission Date: 15/O8/2024
    D1scharge Date: 18/08/2024
    Diagnoisis: Vira1 Fever
    Total Arnount: Rs. 25,000.00
    """
    
    print(f"\n  Input (noisy OCR):\n{noisy_text}")
    
    result = enhancer.enhance_extraction(noisy_text)
    
    print("\n  Enhanced extraction:")
    for field, value in result.items():
        if value and isinstance(value, str):
            print(f"    {field}: {value}")
    
    return len(result) > 0


def test_preprocessing_config():
    """Test preprocessing configuration"""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Config")
    print("=" * 60)
    
    from preprocessing import PreprocessingConfig, ImagePreprocessor
    
    config = PreprocessingConfig(
        dpi=350,
        blur_kernel=(3, 3),
        adaptive_block_size=31,
        min_blob_area=50
    )
    
    print(f"  DPI: {config.dpi}")
    print(f"  Blur kernel: {config.blur_kernel}")
    print(f"  Adaptive block size: {config.adaptive_block_size}")
    print(f"  Min blob area: {config.min_blob_area}")
    
    preprocessor = ImagePreprocessor(config)
    print(f"\n  ImagePreprocessor created successfully")
    
    return True


def test_layout_detection_classes():
    """Test layout detection data classes"""
    print("\n" + "=" * 60)
    print("Testing Layout Detection Classes")
    print("=" * 60)
    
    from layout_detection import FieldType, BoundingBox, FormField
    
    # Test FieldType
    print(f"\n  Field types:")
    for ft in FieldType:
        print(f"    - {ft.value}")
    
    # Test BoundingBox
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    print(f"\n  BoundingBox: {bbox}")
    print(f"    Area: {bbox.area}")
    print(f"    Center: {bbox.center}")
    
    # Test contains with another bbox
    inner_bbox = BoundingBox(x=20, y=30, width=20, height=20)
    print(f"    Contains inner bbox: {bbox.contains(inner_bbox)}")
    
    # Test FormField
    field = FormField(
        bbox=bbox,
        field_type=FieldType.TEXT_PRINTED,
        label="Patient Name"
    )
    print(f"\n  FormField: type={field.field_type.value}, label={field.label}")
    
    return True


def test_field_extractors():
    """Test field extractor configs"""
    print("\n" + "=" * 60)
    print("Testing Field Extractors")
    print("=" * 60)
    
    from field_extractors import TesseractConfig, PrintedTextExtractor
    
    # Test configs
    configs = ['default', 'single_line', 'names', 'policy_number', 'dates', 'numbers_only']
    
    extractor = PrintedTextExtractor()
    
    print("\n  Available Tesseract configs:")
    for name in configs:
        config = extractor.CONFIGS.get(name, TesseractConfig())
        print(f"    {name}: psm={config.psm}, oem={config.oem}, whitelist='{config.whitelist[:20]}...' if config.whitelist else 'all'")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ENHANCED OCR PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Preprocessing Config", test_preprocessing_config),
        ("Layout Detection Classes", test_layout_detection_classes),
        ("Field Extractors", test_field_extractors),
        ("Text Postprocessing", test_text_postprocessing),
        ("OCR Noise Generator", test_ocr_noise_generator),
        ("Hybrid Extractor", test_hybrid_extractor),
        ("OCR Enhancer", test_ocr_enhancer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            logger.error(f"Test '{name}' failed with exception: {e}")
            results.append((name, f"ERROR: {str(e)[:50]}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    print(f"\n  Total: {passed}/{len(results)} tests passed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
