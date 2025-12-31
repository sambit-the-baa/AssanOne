"""
BERT NER Inference Module for OCR Text
- Load trained BERT NER model
- Extract entities from noisy OCR text
- Handle various input formats
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float


class BERTNERExtractor:
    """BERT-based NER for insurance claim extraction"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.id2label = {}
        self.label2id = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load BERT NER model"""
        try:
            import torch
            from transformers import BertTokenizerFast, BertForTokenClassification
            
            logger.info(f"Loading BERT NER model from {model_path}")
            
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
            self.model = BertForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            
            # Get label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            logger.info(f"Model loaded with {len(self.id2label)} labels")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text (can be noisy OCR output)
        
        Returns:
            List of Entity objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0].tolist()
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(predictions, dim=-1)[0].tolist()
            confidences = torch.max(predictions, dim=-1)[0][0].tolist()
        
        # Convert to entities
        entities = []
        current_entity = None
        
        for i, (pred_id, conf, offset) in enumerate(zip(predicted_ids, confidences, offset_mapping)):
            if offset == (0, 0):  # Special token
                continue
            
            label = self.id2label.get(str(pred_id), 'O')
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]
                current_entity = Entity(
                    text=text[offset[0]:offset[1]],
                    label=entity_type,
                    start=offset[0],
                    end=offset[1],
                    confidence=conf
                )
            
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity.label:
                    # Continue entity
                    current_entity.text = text[current_entity.start:offset[1]]
                    current_entity.end = offset[1]
                    current_entity.confidence = min(current_entity.confidence, conf)
                else:
                    # Different type - save and start new
                    entities.append(current_entity)
                    current_entity = Entity(
                        text=text[offset[0]:offset[1]],
                        label=entity_type,
                        start=offset[0],
                        end=offset[1],
                        confidence=conf
                    )
            
            elif label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def extract_to_dict(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and return as structured dictionary.
        Groups entities by type and returns best match for each.
        """
        entities = self.extract_entities(text)
        
        # Group by label
        grouped = {}
        for entity in entities:
            if entity.label not in grouped:
                grouped[entity.label] = []
            grouped[entity.label].append(entity)
        
        # Select best entity for each type (highest confidence)
        result = {}
        for label, entity_list in grouped.items():
            entity_list.sort(key=lambda x: x.confidence, reverse=True)
            best = entity_list[0]
            result[label] = {
                'value': best.text.strip(),
                'confidence': best.confidence,
                'all_matches': [{'text': e.text, 'confidence': e.confidence} for e in entity_list]
            }
        
        return result
    
    def extract_claim_fields(self, text: str) -> Dict[str, str]:
        """
        Extract claim-specific fields and return simple string values.
        Maps to expected field names.
        """
        entities = self.extract_to_dict(text)
        
        # Map to standard field names
        field_mapping = {
            'PATIENT_NAME': 'claimant_name',
            'HOSPITAL': 'hospital_name',
            'POLICY_NUMBER': 'policy_number',
            'CLAIM_ID': 'claim_id',
            'ADMISSION_DATE': 'admission_date',
            'DISCHARGE_DATE': 'discharge_date',
            'DIAGNOSIS': 'diagnosis',
            'AMOUNT': 'amount',
            'DOCTOR_NAME': 'doctor_name',
            'FACILITY_ADDRESS': 'address',
            'PROCEDURE': 'procedure',
            'ICD_CODE': 'icd_code',
        }
        
        result = {}
        for ner_label, field_name in field_mapping.items():
            if ner_label in entities:
                result[field_name] = entities[ner_label]['value']
        
        return result


class HybridExtractor:
    """
    Hybrid extraction combining regex patterns with BERT NER.
    Falls back to regex when NER confidence is low.
    """
    
    def __init__(self, ner_model_path: str = None, confidence_threshold: float = 0.7):
        self.ner_extractor = None
        self.confidence_threshold = confidence_threshold
        
        if ner_model_path and os.path.exists(ner_model_path):
            try:
                self.ner_extractor = BERTNERExtractor(ner_model_path)
            except Exception as e:
                logger.warning(f"Failed to load NER model: {e}. Using regex only.")
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for fallback extraction"""
        self.patterns = {
            'claimant_name': [
                # Patient Name field
                r"(?:Patient|Claimant|Insured|Name)\s*(?:Name)?\s*[:\-]?\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})",
                # ALL CAPS name
                r"(?:Patient|Name)\s*[:\-]?\s*([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})",
                # Name followed by common patterns
                r"(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s*,|\s+(?:S/O|D/O|W/O|Age))",
            ],
            'policy_number': [
                r"(?:Policy|Pol)\.?\s*(?:No|Number|#)?\.?\s*[:\-]?\s*([A-Z]{2,5}[0-9]{6,12})",
                r"(?:Policy|Insurance)\s*(?:ID|Number)\s*[:\-]?\s*([A-Z0-9]{8,15})",
            ],
            'hospital_name': [
                r"(?:Hospital|Facility|Provider)\s*(?:Name)?\s*[:\-]?\s*([A-Z][A-Za-z\s]+(?:Hospital|Medical|Healthcare|Clinic))",
                r"([A-Z][A-Za-z\s]+(?:Hospital|Medical Center|Healthcare|Clinic))",
            ],
            'admission_date': [
                r"(?:Admission|Admit|DOA)\s*(?:Date)?\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"(?:Date\s+of\s+Admission)\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
            ],
            'discharge_date': [
                r"(?:Discharge|DOD)\s*(?:Date)?\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"(?:Date\s+of\s+Discharge)\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
            ],
            'diagnosis': [
                r"(?:Diagnosis|Disease|Condition)\s*[:\-]?\s*([A-Za-z][A-Za-z\s,]+)",
                r"(?:Primary\s+)?Diagnosis\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)",
            ],
            'amount': [
                r"(?:Total|Claim|Bill)\s*(?:Amount)?\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)",
                r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{2})?)",
            ],
        }
        
        # Compile all patterns
        self.compiled_patterns = {}
        for field, patterns in self.patterns.items():
            self.compiled_patterns[field] = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    
    def extract_with_regex(self, text: str, field: str) -> Optional[str]:
        """Extract field using regex patterns"""
        if field not in self.compiled_patterns:
            return None
        
        for pattern in self.compiled_patterns[field]:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all fields using hybrid approach.
        Uses NER first, falls back to regex for low-confidence results.
        """
        result = {}
        
        # Try NER extraction first
        if self.ner_extractor:
            try:
                ner_result = self.ner_extractor.extract_to_dict(text)
                
                for label, data in ner_result.items():
                    field_name = self._label_to_field(label)
                    if field_name:
                        if data['confidence'] >= self.confidence_threshold:
                            result[field_name] = {
                                'value': data['value'],
                                'source': 'ner',
                                'confidence': data['confidence']
                            }
                        else:
                            # Low confidence - try regex
                            regex_value = self.extract_with_regex(text, field_name)
                            if regex_value:
                                result[field_name] = {
                                    'value': regex_value,
                                    'source': 'regex',
                                    'confidence': 0.5  # Fixed confidence for regex
                                }
                            else:
                                # Use NER result anyway
                                result[field_name] = {
                                    'value': data['value'],
                                    'source': 'ner_low_conf',
                                    'confidence': data['confidence']
                                }
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}. Using regex only.")
        
        # Fill missing fields with regex
        for field in self.patterns.keys():
            if field not in result:
                regex_value = self.extract_with_regex(text, field)
                if regex_value:
                    result[field] = {
                        'value': regex_value,
                        'source': 'regex',
                        'confidence': 0.5
                    }
        
        return result
    
    def _label_to_field(self, label: str) -> Optional[str]:
        """Convert NER label to field name"""
        mapping = {
            'PATIENT_NAME': 'claimant_name',
            'HOSPITAL': 'hospital_name',
            'POLICY_NUMBER': 'policy_number',
            'CLAIM_ID': 'claim_id',
            'ADMISSION_DATE': 'admission_date',
            'DISCHARGE_DATE': 'discharge_date',
            'DIAGNOSIS': 'diagnosis',
            'AMOUNT': 'amount',
            'DOCTOR_NAME': 'doctor_name',
            'FACILITY_ADDRESS': 'address',
        }
        return mapping.get(label)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ocr_training/models/bert_ner",
                        help="Path to BERT NER model")
    parser.add_argument("--text", help="Text to extract entities from")
    
    args = parser.parse_args()
    
    # Test with sample text
    sample_text = """
    Patient Name: Sandhya Raghunath Kate
    Policy No: ABC12345678
    Hospital: ACCIDENT HOSPITAL VADUJ
    Admission Date: 15/08/2024
    Discharge Date: 18/08/2024
    Diagnosis: Viral Fever
    Total Amount: Rs. 25,000.00
    """
    
    text_to_process = args.text or sample_text
    
    print("=" * 60)
    print("Testing Hybrid Extractor")
    print("=" * 60)
    
    extractor = HybridExtractor(args.model if os.path.exists(args.model) else None)
    result = extractor.extract(text_to_process)
    
    print(f"\nInput Text:\n{text_to_process}")
    print("\nExtracted Fields:")
    for field, data in result.items():
        print(f"  {field}: {data['value']} (source: {data['source']}, conf: {data['confidence']:.2f})")
