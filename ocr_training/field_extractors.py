"""
Specialized Field Extractors
- Tesseract with tuned config for printed fields
- Handwriting recognition for handwritten fields
- Checkbox classifier for box states
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

from layout_detection import FormField, FieldType, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class TesseractConfig:
    """Tesseract OCR configuration"""
    # Page segmentation modes:
    # 0 = OSD only, 1 = Auto+OSD, 2 = Auto no OSD, 3 = Fully auto
    # 4 = Single column, 5 = Single uniform block (vertical), 6 = Single uniform block
    # 7 = Single text line, 8 = Single word, 9 = Single word in circle
    # 10 = Single character, 11 = Sparse text, 12 = Sparse text + OSD, 13 = Raw line
    psm: int = 6
    
    # OCR Engine modes:
    # 0 = Legacy only, 1 = LSTM only, 2 = Legacy + LSTM, 3 = Default
    oem: int = 3
    
    # Language (eng, hin, mar for Indian languages)
    lang: str = "eng"
    
    # Character whitelist (empty = all characters)
    whitelist: str = ""
    
    # Additional config options
    extra_config: str = ""
    
    def to_string(self) -> str:
        """Convert to Tesseract config string"""
        config_parts = [f"--psm {self.psm}", f"--oem {self.oem}"]
        
        if self.whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={self.whitelist}")
        
        if self.extra_config:
            config_parts.append(self.extra_config)
        
        return " ".join(config_parts)


class PrintedTextExtractor:
    """Optimized Tesseract OCR for printed text"""
    
    # Pre-defined configurations for different field types
    CONFIGS = {
        'default': TesseractConfig(psm=6, oem=3),
        'single_line': TesseractConfig(psm=7, oem=3),
        'single_word': TesseractConfig(psm=8, oem=3),
        'sparse': TesseractConfig(psm=11, oem=3),
        'numbers_only': TesseractConfig(psm=7, oem=3, whitelist="0123456789.-,"),
        'alphanumeric': TesseractConfig(psm=7, oem=3, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
        'dates': TesseractConfig(psm=7, oem=3, whitelist="0123456789/-"),
        'currency': TesseractConfig(psm=7, oem=3, whitelist="0123456789.,₹$RsINR "),
        'names': TesseractConfig(psm=7, oem=3, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. "),
        'policy_number': TesseractConfig(psm=8, oem=3, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"),
    }
    
    def __init__(self, lang: str = "eng"):
        self.lang = lang
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Additional preprocessing for printed text OCR"""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small
        h, w = gray.shape[:2]
        if h < 30:
            scale = 30 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Binarize using Otsu's method
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure text is black on white background
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def extract(self, image: np.ndarray, config_name: str = 'default') -> Dict[str, Any]:
        """
        Extract text from printed region.
        
        Args:
            image: Input image (cropped field region)
            config_name: Name of predefined config to use
        
        Returns:
            Dictionary with extracted text and confidence
        """
        # Get configuration
        config = self.CONFIGS.get(config_name, self.CONFIGS['default'])
        config.lang = self.lang
        
        # Preprocess
        processed = self.preprocess_for_ocr(image)
        
        # Run Tesseract with detailed output
        try:
            data = pytesseract.image_to_data(
                processed,
                lang=config.lang,
                config=config.to_string(),
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for i, text in enumerate(data['text']):
                if text.strip():
                    texts.append(text)
                    conf = int(data['conf'][i])
                    if conf > 0:
                        confidences.append(conf)
            
            full_text = " ".join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence / 100,
                'word_count': len(texts),
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {'text': '', 'confidence': 0, 'error': str(e)}
    
    def extract_with_multiple_configs(self, image: np.ndarray, 
                                       configs: List[str] = None) -> Dict[str, Any]:
        """Try multiple configurations and return best result"""
        if configs is None:
            configs = ['default', 'single_line', 'sparse']
        
        best_result = {'text': '', 'confidence': 0}
        
        for config_name in configs:
            result = self.extract(image, config_name)
            if result['confidence'] > best_result['confidence']:
                best_result = result
                best_result['config_used'] = config_name
        
        return best_result


class HandwritingExtractor:
    """
    Handwriting recognition using:
    1. Enhanced Tesseract with handwriting-optimized settings
    2. Optional deep learning model (TrOCR, etc.)
    """
    
    def __init__(self, use_deep_model: bool = False):
        self.use_deep_model = use_deep_model
        self.deep_model = None
        
        if use_deep_model:
            self._load_deep_model()
    
    def _load_deep_model(self):
        """Load TrOCR or similar handwriting model"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            logger.info("Loading TrOCR handwriting model...")
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.deep_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            logger.info("TrOCR model loaded successfully")
            
        except ImportError:
            logger.warning("transformers not installed, falling back to Tesseract")
            self.use_deep_model = False
        except Exception as e:
            logger.warning(f"Failed to load TrOCR model: {e}")
            self.use_deep_model = False
    
    def preprocess_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for handwritten text"""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to optimal height for handwriting
        h, w = gray.shape[:2]
        target_height = 64
        if h != target_height:
            scale = target_height / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding (better for varying ink density)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        
        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract handwriting using Tesseract"""
        processed = self.preprocess_handwriting(image)
        
        # Tesseract config for handwriting
        config = "--psm 7 --oem 3 -c tessedit_do_invert=0"
        
        try:
            text = pytesseract.image_to_string(processed, lang='eng', config=config)
            
            # Get confidence
            data = pytesseract.image_to_data(
                processed, lang='eng', config=config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_conf = np.mean(confidences) / 100 if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_conf,
                'method': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Handwriting extraction failed: {e}")
            return {'text': '', 'confidence': 0, 'error': str(e)}
    
    def extract_deep(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract handwriting using deep learning model"""
        if not self.use_deep_model or self.deep_model is None:
            return self.extract_tesseract(image)
        
        try:
            from PIL import Image
            
            # Convert to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process with TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            generated_ids = self.deep_model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                'text': text,
                'confidence': 0.8,  # TrOCR doesn't provide confidence
                'method': 'trocr'
            }
            
        except Exception as e:
            logger.error(f"Deep model extraction failed: {e}, falling back to Tesseract")
            return self.extract_tesseract(image)
    
    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract handwritten text"""
        if self.use_deep_model:
            return self.extract_deep(image)
        else:
            return self.extract_tesseract(image)


class CheckboxClassifier:
    """Classify checkbox states: checked, unchecked, or indeterminate"""
    
    def __init__(self, 
                 fill_threshold: float = 0.25,
                 checkmark_threshold: float = 0.15):
        self.fill_threshold = fill_threshold
        self.checkmark_threshold = checkmark_threshold
    
    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify checkbox state.
        
        Returns:
            Dictionary with state, confidence, and analysis details
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to standard size for consistent analysis
        standard_size = (32, 32)
        resized = cv2.resize(gray, standard_size, interpolation=cv2.INTER_AREA)
        
        # Binarize
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate fill ratio
        total_pixels = binary.size
        dark_pixels = np.sum(binary > 0)
        fill_ratio = dark_pixels / total_pixels
        
        # Check for X or checkmark patterns
        has_checkmark = self._detect_checkmark(binary)
        has_x_mark = self._detect_x_mark(binary)
        
        # Determine state
        if fill_ratio > self.fill_threshold or has_checkmark or has_x_mark:
            state = "checked"
            confidence = min(0.95, 0.5 + fill_ratio + (0.2 if has_checkmark or has_x_mark else 0))
        elif fill_ratio < 0.05:
            state = "unchecked"
            confidence = 0.9
        else:
            state = "indeterminate"
            confidence = 0.5
        
        return {
            'state': state,
            'is_checked': state == "checked",
            'confidence': confidence,
            'fill_ratio': fill_ratio,
            'has_checkmark': has_checkmark,
            'has_x_mark': has_x_mark
        }
    
    def _detect_checkmark(self, binary: np.ndarray) -> bool:
        """Detect checkmark pattern (✓)"""
        # Look for diagonal lines that form a checkmark
        h, w = binary.shape
        
        # Check lower-left to center and center to upper-right pattern
        center = (w // 2, h // 2)
        
        # Simple heuristic: check diagonal pixel density
        diag1_sum = 0  # Lower-left diagonal
        diag2_sum = 0  # Upper-right diagonal
        
        for i in range(min(h, w) // 2):
            # Lower-left to center
            if h - 1 - i >= 0 and i < w // 2:
                diag1_sum += binary[h - 1 - i, i]
            # Center to upper-right
            if center[1] - i >= 0 and center[0] + i < w:
                diag2_sum += binary[center[1] - i, center[0] + i]
        
        return diag1_sum > 100 and diag2_sum > 50
    
    def _detect_x_mark(self, binary: np.ndarray) -> bool:
        """Detect X mark pattern"""
        h, w = binary.shape
        
        # Check both diagonals
        diag1_sum = 0
        diag2_sum = 0
        
        for i in range(min(h, w)):
            if i < h and i < w:
                diag1_sum += binary[i, i]  # Top-left to bottom-right
                diag2_sum += binary[i, w - 1 - i]  # Top-right to bottom-left
        
        threshold = min(h, w) * 50  # Adjust based on expected ink density
        return diag1_sum > threshold and diag2_sum > threshold


class FieldExtractorPipeline:
    """
    Combined pipeline for extracting all field types.
    Routes each field to the appropriate extractor based on FieldType.
    """
    
    def __init__(self, lang: str = "eng", use_handwriting_model: bool = False):
        self.printed_extractor = PrintedTextExtractor(lang)
        self.handwriting_extractor = HandwritingExtractor(use_handwriting_model)
        self.checkbox_classifier = CheckboxClassifier()
    
    def extract_field(self, field: FormField, image: np.ndarray) -> FormField:
        """
        Extract content from a single field.
        Updates field.extracted_text and field.metadata.
        """
        # Crop field region if not already done
        if field.content_image is None:
            field.content_image = field.bbox.crop(image)
        
        roi = field.content_image
        
        if field.field_type == FieldType.CHECKBOX:
            result = self.checkbox_classifier.classify(roi)
            field.extracted_text = "checked" if result['is_checked'] else "unchecked"
            field.confidence = result['confidence']
            field.metadata.update(result)
            
        elif field.field_type == FieldType.TEXT_HANDWRITTEN:
            result = self.handwriting_extractor.extract(roi)
            field.extracted_text = result['text']
            field.confidence = result['confidence']
            field.metadata['extraction_method'] = result.get('method', 'unknown')
            
        elif field.field_type in [FieldType.TEXT_PRINTED, FieldType.LABEL]:
            result = self.printed_extractor.extract_with_multiple_configs(roi)
            field.extracted_text = result['text']
            field.confidence = result['confidence']
            field.metadata['config_used'] = result.get('config_used', 'default')
            
        elif field.field_type == FieldType.TABLE_CELL:
            # Try both printed and check if looks like number
            result = self.printed_extractor.extract(roi, 'default')
            
            # If mostly numbers, re-extract with numbers config
            if result['text'] and re.match(r'^[\d\s.,/-]+$', result['text']):
                result = self.printed_extractor.extract(roi, 'numbers_only')
            
            field.extracted_text = result['text']
            field.confidence = result['confidence']
            
        else:
            # Unknown type - try printed text extraction
            result = self.printed_extractor.extract(roi, 'default')
            field.extracted_text = result['text']
            field.confidence = result['confidence']
        
        return field
    
    def extract_all_fields(self, fields: List[FormField], 
                           image: np.ndarray) -> List[FormField]:
        """Extract content from all fields"""
        logger.info(f"Extracting {len(fields)} fields")
        
        for i, field in enumerate(fields):
            try:
                self.extract_field(field, image)
                logger.debug(f"Field {i+1}: {field.field_type.value} -> '{field.extracted_text[:50]}...' "
                            f"(conf: {field.confidence:.2f})")
            except Exception as e:
                logger.error(f"Failed to extract field {i+1}: {e}")
                field.extracted_text = ""
                field.confidence = 0
                field.metadata['error'] = str(e)
        
        return fields
    
    def extract_specific_field(self, image: np.ndarray, 
                               field_type: str,
                               config: str = None) -> Dict[str, Any]:
        """
        Extract text from image using specific field type settings.
        
        Args:
            image: Input image
            field_type: One of 'name', 'policy_number', 'date', 'currency', 'number'
            config: Optional override config name
        """
        config_map = {
            'name': 'names',
            'policy_number': 'policy_number',
            'date': 'dates',
            'currency': 'currency',
            'number': 'numbers_only',
            'general': 'default'
        }
        
        config_name = config or config_map.get(field_type, 'default')
        return self.printed_extractor.extract(image, config_name)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python field_extractors.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    # Test printed text extraction
    extractor = PrintedTextExtractor()
    result = extractor.extract_with_multiple_configs(image)
    
    print(f"Extracted text: {result['text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Config used: {result.get('config_used', 'unknown')}")
