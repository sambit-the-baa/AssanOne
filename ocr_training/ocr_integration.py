"""
OCR Enhancement Integration
============================
Integrates trained OCR models with the fraud detection pipeline
to enhance document extraction accuracy and reduce gibberish detection.

This module provides:
1. ML-enhanced entity extraction from documents
2. Fallback handling when ML models are unavailable
3. Integration points for the existing extractor.py

Usage:
    from ocr_integration import OCREnhancer
    
    enhancer = OCREnhancer()
    enhanced_data = enhancer.enhance_extraction(raw_text, existing_data)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class OCREnhancer:
    """
    Enhances OCR extraction using ML models when available,
    with graceful fallback to rule-based extraction.
    """
    
    # Patterns for common OCR errors in insurance documents
    OCR_ERROR_PATTERNS = {
        # Common character substitutions
        '0': ['O', 'o', 'Q'],
        '1': ['l', 'I', '|', 'i'],
        '5': ['S', 's'],
        '8': ['B'],
        '6': ['G', 'b'],
        '2': ['Z', 'z'],
        # Common word errors
        'hospital': ['hospita1', 'hosp1tal', 'hospltal'],
        'patient': ['pat1ent', 'patlent'],
        'diagnosis': ['diagnos1s', 'd1agnosis'],
        'amount': ['amounl', 'am0unt'],
        'total': ['tota1', 't0tal'],
    }
    
    # Insurance-specific field patterns
    INSURANCE_PATTERNS = {
        'claim_id': [
            r'claim\s*(?:no|number|id)?[:\s]*([A-Z0-9-]+)',
            r'(?:TPA|claim)\s*ref[:\s]*([A-Z0-9-]+)',
        ],
        'policy_number': [
            r'policy\s*(?:no|number)?[:\s]*([A-Z0-9-]+)',
            r'polic[ey]\s*[:\s]*([A-Z0-9-]+)',
        ],
        'hospital_name': [
            r'hospital[:\s]*([A-Za-z\s]+(?:Hospital|Medical|Healthcare|Clinic))',
            r'([A-Za-z\s]+(?:Hospital|Medical Centre|Health Care|Nursing Home))',
        ],
        'patient_name': [
            r'patient\s*name\s*[:\s]+([A-Za-z\s.]+?)(?:\n|Claim|Policy|$)',
            r'name\s*of\s*patient[:\s]+([A-Za-z\s.]+?)(?:\n|$)',
            r'patient[:\s]+([A-Za-z]+\s+[A-Za-z]+)(?:\n|$)',
        ],
        'admission_date': [
            r'(?:date\s*of\s*)?admission[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'admitted\s*(?:on)?[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ],
        'discharge_date': [
            r'(?:date\s*of\s*)?discharge[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'discharged\s*(?:on)?[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ],
        'total_amount': [
            r'total\s*(?:amount)?[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:grand\s*)?total[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{2})?)\s*(?:only)?$',
        ],
        'diagnosis': [
            r'diagnosis[:\s]*([A-Za-z\s,]+?)(?:\n|$)',
            r'provisional\s*diagnosis[:\s]*([A-Za-z\s,]+?)(?:\n|$)',
            r'final\s*diagnosis[:\s]*([A-Za-z\s,]+?)(?:\n|$)',
        ],
        'icd_code': [
            r'ICD[:\s-]*10?[:\s-]*([A-Z]\d{2}(?:\.\d{1,2})?)',
            r'([A-Z]\d{2}\.\d{1,2})',  # ICD format like J18.9
        ],
        'procedure': [
            r'procedure[:\s]*([A-Za-z\s,]+?)(?:\n|$)',
            r'surgery[:\s]*([A-Za-z\s,]+?)(?:\n|$)',
            r'operation[:\s]*([A-Za-z\s,]+?)(?:\n|$)',
        ],
    }
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize OCR enhancer.
        
        Args:
            model_dir: Directory containing trained ML models (optional)
        """
        # Auto-detect model directory if not provided
        if model_dir is None:
            # Check relative to this file
            default_model_dir = Path(__file__).parent / "models"
            if default_model_dir.exists():
                model_dir = str(default_model_dir)
        
        self.model_dir = Path(model_dir) if model_dir else None
        self.ml_model = None
        self.ml_tokenizer = None
        self.ml_available = False
        self.id2label = {}
        self.device = "cpu"
        
        # Try to load ML model if available
        if self.model_dir and self.model_dir.exists():
            self._try_load_ml_model()
    
    def _try_load_ml_model(self):
        """Try to load ML model for enhanced extraction."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            
            model_path = self.model_dir / "bert_ner"
            if model_path.exists():
                self.ml_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.ml_model = AutoModelForTokenClassification.from_pretrained(model_path)
                
                # Get label mapping from model config
                self.id2label = self.ml_model.config.id2label
                
                # Move to GPU if available
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.ml_model = self.ml_model.to(self.device)
                self.ml_model.eval()  # Set to evaluation mode
                
                self.ml_available = True
                logger.info(f"Loaded ML model from {model_path} on {self.device}")
        except ImportError:
            logger.debug("transformers not available, using rule-based extraction only")
        except Exception as e:
            logger.debug(f"Could not load ML model: {e}")
    
    def enhance_extraction(
        self, 
        raw_text: str, 
        existing_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance extraction from raw OCR text.
        
        Args:
            raw_text: Raw OCR text from document
            existing_data: Previously extracted data (optional)
        
        Returns:
            Enhanced extraction with all detected fields
        """
        # Start with existing data or empty dict
        result = existing_data.copy() if existing_data else {}
        
        # Clean OCR text
        cleaned_text = self._clean_ocr_text(raw_text)
        
        # Extract using patterns
        pattern_extractions = self._extract_with_patterns(cleaned_text)
        
        # Merge extractions (pattern fills gaps)
        for field, value in pattern_extractions.items():
            if value and (not result.get(field) or self._is_low_quality(result.get(field, ''))):
                result[field] = value
        
        # Try ML extraction if available
        if self.ml_available:
            ml_extractions = self._extract_with_ml(cleaned_text)
            for field, value in ml_extractions.items():
                if value and (not result.get(field) or self._is_low_quality(result.get(field, ''))):
                    result[field] = value
        
        # Post-process and validate
        result = self._post_process(result)
        
        return result
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean common OCR errors from text."""
        if not text:
            return ""
        
        cleaned = text
        
        # Fix common OCR substitutions in keywords
        for correct, errors in self.OCR_ERROR_PATTERNS.items():
            if isinstance(errors, list):
                for error in errors:
                    # Only replace in specific contexts (near keywords)
                    cleaned = re.sub(
                        rf'(?i)({error})(?=\s*(?:hospital|patient|amount|total|diagnosis))',
                        correct,
                        cleaned
                    )
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common character replacements
        cleaned = cleaned.replace('|', 'I')  # Pipe often misread as I
        
        return cleaned.strip()
    
    def _extract_with_patterns(self, text: str) -> Dict[str, str]:
        """Extract fields using regex patterns."""
        extractions = {}
        
        for field, patterns in self.INSURANCE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and not self._is_low_quality(value):
                        extractions[field] = value
                        break
        
        return extractions
    
    def _extract_with_ml(self, text: str) -> Dict[str, Any]:
        """Extract entities using ML model."""
        if not self.ml_available or not self.ml_model:
            return {}
        
        try:
            import torch
            
            # Tokenize the text
            inputs = self.ml_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                return_offsets_mapping=True
            )
            
            # Get offset mapping before moving to device
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
            
            predictions = outputs.logits.argmax(dim=2)[0].cpu().tolist()
            
            # Parse predictions into entities
            entities = self._parse_ner_predictions(text, predictions, offset_mapping)
            
            # Convert entities to field dictionary
            extractions = {}
            for entity_type, entity_values in entities.items():
                if entity_values:
                    # Map entity type to field name
                    field_name = entity_type.lower()
                    # Take the first (most confident) value
                    extractions[field_name] = entity_values[0]
            
            return extractions
            
        except Exception as e:
            logger.debug(f"ML extraction failed: {e}")
            return {}
    
    def _parse_ner_predictions(
        self, 
        text: str, 
        predictions: List[int], 
        offset_mapping: List[Tuple[int, int]]
    ) -> Dict[str, List[str]]:
        """Parse NER predictions into entity dictionary."""
        entities = {}
        current_entity = None
        current_tokens = []
        
        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Skip special tokens
                continue
            
            label = self.id2label.get(pred, "O")
            
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_tokens:
                    entity_text = self._reconstruct_entity(text, current_tokens)
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append(entity_text)
                
                # Start new entity
                current_entity = label[2:]  # Remove "B-" prefix
                current_tokens = [(start, end)]
                
            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]  # Remove "I-" prefix
                if entity_type == current_entity:
                    current_tokens.append((start, end))
                else:
                    # Entity type mismatch, save current and start new
                    if current_tokens:
                        entity_text = self._reconstruct_entity(text, current_tokens)
                        if current_entity not in entities:
                            entities[current_entity] = []
                        entities[current_entity].append(entity_text)
                    current_entity = entity_type
                    current_tokens = [(start, end)]
            else:
                # O label - save current entity if exists
                if current_entity and current_tokens:
                    entity_text = self._reconstruct_entity(text, current_tokens)
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append(entity_text)
                current_entity = None
                current_tokens = []
        
        # Don't forget the last entity
        if current_entity and current_tokens:
            entity_text = self._reconstruct_entity(text, current_tokens)
            if current_entity not in entities:
                entities[current_entity] = []
            entities[current_entity].append(entity_text)
        
        return entities
    
    def _reconstruct_entity(self, text: str, token_spans: List[Tuple[int, int]]) -> str:
        """Reconstruct entity text from token spans."""
        if not token_spans:
            return ""
        
        # Get the full span from first to last token
        start = token_spans[0][0]
        end = token_spans[-1][1]
        
        return text[start:end].strip()
    
    def _is_low_quality(self, text: str) -> bool:
        """Check if extracted text is low quality/gibberish."""
        if not text or len(str(text).strip()) < 2:
            return True
        
        text_str = str(text)
        
        # Too short
        if len(text_str) < 2:
            return True
        
        # Too many special characters
        special_ratio = sum(1 for c in text_str if not c.isalnum() and c != ' ') / len(text_str)
        if special_ratio > 0.4:
            return True
        
        # No vowels in alpha text (likely gibberish)
        alpha_chars = [c for c in text_str if c.isalpha()]
        if alpha_chars:
            vowels = set('aeiouAEIOU')
            if not any(c in vowels for c in alpha_chars):
                return True
        
        # Repeating characters
        if re.search(r'(.)\1{4,}', text_str):  # 5+ repeating chars
            return True
        
        # Common form template text
        form_patterns = [
            r'^date\s*$', r'^name\s*$', r'^address\s*$',
            r'^_+$', r'^\.+$', r'^\s*$',
            r'^please\s+fill', r'^enter\s+your',
        ]
        for pattern in form_patterns:
            if re.match(pattern, text_str.lower().strip()):
                return True
        
        return False
    
    def _post_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extracted data for consistency."""
        processed = data.copy()
        
        # Clean amount fields
        if 'total_amount' in processed:
            amount = processed['total_amount']
            if isinstance(amount, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r'[Rs₹,\s]', '', amount)
                try:
                    processed['total_amount'] = float(cleaned)
                except ValueError:
                    pass
        
        # Standardize date formats
        date_fields = ['admission_date', 'discharge_date', 'service_date']
        for field in date_fields:
            if field in processed:
                processed[field] = self._standardize_date(processed[field])
        
        # Clean names
        name_fields = ['patient_name', 'hospital_name', 'doctor_name']
        for field in name_fields:
            if field in processed:
                processed[field] = self._clean_name(processed[field])
        
        # Validate ICD codes
        if 'icd_code' in processed:
            icd = processed['icd_code']
            if not re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', str(icd)):
                del processed['icd_code']
        
        return processed
    
    def _standardize_date(self, date_str: str) -> str:
        """Standardize date format to DD/MM/YYYY."""
        if not date_str:
            return date_str
        
        # Try to parse common formats
        patterns = [
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', r'\1/\2/\3'),
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})', r'\1/\2/20\3'),
        ]
        
        for pattern, replacement in patterns:
            match = re.match(pattern, date_str)
            if match:
                return re.sub(pattern, replacement, date_str)
        
        return date_str
    
    def _clean_name(self, name: str) -> str:
        """Clean and standardize names."""
        if not name:
            return name
        
        # Remove extra whitespace
        cleaned = ' '.join(name.split())
        
        # Title case
        cleaned = cleaned.title()
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[_|]', '', cleaned)
        
        return cleaned.strip()


def integrate_with_extractor():
    """
    Example integration with existing extractor.py
    
    Add this to the extraction flow:
    
    from ocr_integration import OCREnhancer
    
    enhancer = OCREnhancer(model_dir="ocr_training/models")
    
    # After initial extraction
    enhanced_data = enhancer.enhance_extraction(
        raw_text=tesseract_output,
        existing_data=initial_extraction
    )
    """
    pass


# Quick test
if __name__ == "__main__":
    # Test the enhancer
    print("Initializing OCR Enhancer...")
    enhancer = OCREnhancer()
    
    print(f"ML Model Available: {enhancer.ml_available}")
    if enhancer.ml_available:
        print(f"Model Device: {enhancer.device}")
        print(f"Labels: {list(enhancer.id2label.values())[:10]}...")
    
    sample_text = """
    Apollo Hospital
    Patient Name: John Doe
    Claim No: CLM123456
    Policy Number: POL789012
    Date of Admission: 15/01/2024
    Date of Discharge: 20/01/2024
    Diagnosis: Acute Appendicitis
    Procedure: Laparoscopic Appendectomy
    ICD Code: K35.80
    Total Amount: Rs. 45,000.00
    Doctor: Dr. Rajesh Kumar
    """
    
    print("\n" + "="*50)
    print("Sample Text:")
    print(sample_text)
    print("="*50)
    
    result = enhancer.enhance_extraction(sample_text)
    print("\nExtracted fields:")
    for field, value in sorted(result.items()):
        print(f"  {field}: {value}")
