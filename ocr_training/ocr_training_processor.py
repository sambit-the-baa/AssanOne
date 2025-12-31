"""
OCR Training Data Processor
============================
Parses downloaded Hugging Face datasets (FUNSD, SROIE, DocVQA) 
and prepares them for fine-tuning OCR models for insurance document processing.

Usage:
    python ocr_training_processor.py
"""

import json
import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OCRTrainingExample:
    """Standard format for OCR training data."""
    id: str
    source_dataset: str
    text: str
    words: List[str]
    bboxes: List[List[int]]
    labels: Optional[List[int]] = None
    ner_tags: Optional[List[str]] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    question: Optional[str] = None  # For DocVQA
    answer: Optional[str] = None    # For DocVQA
    document_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OCRDatasetParser:
    """Base class for parsing OCR datasets."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
    
    def parse_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        """Parse JSONL file and extract RECORD entries."""
        filepath = self.data_dir / filename
        records = []
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return records
        
        logger.info(f"Parsing {filename}...")
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            logger.error(f"Could not decode {filename} with any encoding")
            return records
        
        for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('type') == 'RECORD':
                        record_data = data.get('record', {}).get('data', {})
                        if record_data.get('row'):
                            records.append(record_data['row'])
                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping invalid JSON line: {e}")
                    continue
        
        logger.info(f"Parsed {len(records)} records from {filename}")
        return records


class FUNSDParser(OCRDatasetParser):
    """Parser for FUNSD dataset (Form Understanding in Noisy Scanned Documents)."""
    
    # NER tag mapping for FUNSD
    NER_TAG_LABELS = {
        0: 'O',           # Outside
        1: 'B-HEADER',    # Begin header
        2: 'I-HEADER',    # Inside header
        3: 'B-QUESTION',  # Begin question
        4: 'I-QUESTION',  # Inside question
        5: 'B-ANSWER',    # Begin answer
        6: 'I-ANSWER',    # Inside answer
        7: 'B-OTHER',     # Begin other
        8: 'I-OTHER'      # Inside other
    }
    
    def parse(self) -> List[OCRTrainingExample]:
        """Parse FUNSD dataset."""
        records = self.parse_jsonl("funsd_output.jsonl")
        examples = []
        
        for record in records:
            try:
                example = self._convert_record(record)
                if example:
                    examples.append(example)
            except Exception as e:
                logger.warning(f"Error converting FUNSD record: {e}")
                continue
        
        logger.info(f"Converted {len(examples)} FUNSD examples")
        return examples
    
    def _convert_record(self, record: Dict[str, Any]) -> Optional[OCRTrainingExample]:
        """Convert a FUNSD record to standard format."""
        words = record.get('words', [])
        bboxes = record.get('bboxes', [])
        ner_tags = record.get('ner_tags', [])
        image = record.get('image', {})
        
        if not words or not bboxes:
            return None
        
        # Convert NER tags to labels
        ner_labels = [self.NER_TAG_LABELS.get(tag, 'O') for tag in ner_tags]
        
        # Get image URL
        image_url = image.get('src') if isinstance(image, dict) else None
        
        # Generate unique ID
        text = ' '.join(words)
        record_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        return OCRTrainingExample(
            id=f"funsd_{record_id}",
            source_dataset="FUNSD",
            text=text,
            words=words,
            bboxes=bboxes,
            labels=ner_tags,
            ner_tags=ner_labels,
            image_url=image_url,
            document_type="form",
            metadata={
                "dataset_split": record.get('id', ''),
                "width": image.get('width') if isinstance(image, dict) else None,
                "height": image.get('height') if isinstance(image, dict) else None
            }
        )


class SROIEParser(OCRDatasetParser):
    """Parser for SROIE dataset (Scanned Receipts OCR and Information Extraction)."""
    
    # NER tag mapping for SROIE
    NER_TAG_LABELS = {
        0: 'O',           # Other
        1: 'B-COMPANY',   # Company name
        2: 'I-COMPANY',
        3: 'B-DATE',      # Date
        4: 'I-DATE',
        5: 'B-ADDRESS',   # Address
        6: 'I-ADDRESS',
        7: 'B-TOTAL',     # Total amount
        8: 'I-TOTAL'
    }
    
    def parse(self) -> List[OCRTrainingExample]:
        """Parse SROIE dataset."""
        records = self.parse_jsonl("sroie_output.jsonl")
        examples = []
        
        for record in records:
            try:
                example = self._convert_record(record)
                if example:
                    examples.append(example)
            except Exception as e:
                logger.warning(f"Error converting SROIE record: {e}")
                continue
        
        logger.info(f"Converted {len(examples)} SROIE examples")
        return examples
    
    def _convert_record(self, record: Dict[str, Any]) -> Optional[OCRTrainingExample]:
        """Convert a SROIE record to standard format."""
        words = record.get('words', [])
        bboxes = record.get('bboxes', [])
        ner_tags = record.get('ner_tags', [])
        image_path = record.get('image_path', '')
        
        if not words or not bboxes:
            return None
        
        # Convert NER tags to labels
        ner_labels = [self.NER_TAG_LABELS.get(tag, 'O') for tag in ner_tags]
        
        # Generate unique ID
        text = ' '.join(words)
        record_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        return OCRTrainingExample(
            id=f"sroie_{record_id}",
            source_dataset="SROIE",
            text=text,
            words=words,
            bboxes=bboxes,
            labels=ner_tags,
            ner_tags=ner_labels,
            image_path=image_path,
            document_type="receipt",
            metadata={
                "original_id": record.get('id', ''),
                "config": record.get('config', '')
            }
        )


class DocVQAParser(OCRDatasetParser):
    """Parser for DocVQA dataset (Document Visual Question Answering)."""
    
    def parse(self) -> List[OCRTrainingExample]:
        """Parse DocVQA dataset."""
        records = self.parse_jsonl("docvqa_output.jsonl")
        examples = []
        
        for record in records:
            try:
                example = self._convert_record(record)
                if example:
                    examples.append(example)
            except Exception as e:
                logger.warning(f"Error converting DocVQA record: {e}")
                continue
        
        logger.info(f"Converted {len(examples)} DocVQA examples")
        return examples
    
    def _convert_record(self, record: Dict[str, Any]) -> Optional[OCRTrainingExample]:
        """Convert a DocVQA record to standard format."""
        question = record.get('question', '')
        answers = record.get('answers', [])
        image = record.get('image', {})
        question_types = record.get('question_types', [])
        
        if not question or not answers:
            return None
        
        # Get primary answer
        answer = answers[0] if answers else ''
        
        # Get image URL
        image_url = image.get('src') if isinstance(image, dict) else None
        
        # Generate unique ID
        question_id = record.get('questionId', '')
        record_id = question_id or hashlib.md5(question.encode()).hexdigest()[:12]
        
        return OCRTrainingExample(
            id=f"docvqa_{record_id}",
            source_dataset="DocVQA",
            text=question,  # Question as text
            words=question.split(),  # Tokenize question
            bboxes=[],  # DocVQA doesn't provide word bboxes
            image_url=image_url,
            question=question,
            answer=answer,
            document_type="document_qa",
            metadata={
                "question_types": question_types,
                "all_answers": answers,
                "doc_id": record.get('docId'),
                "ucsf_document_id": record.get('ucsf_document_id'),
                "page_no": record.get('ucsf_document_page_no'),
                "width": image.get('width') if isinstance(image, dict) else None,
                "height": image.get('height') if isinstance(image, dict) else None
            }
        )


class InsuranceOCRDataGenerator:
    """
    Generate insurance-specific OCR training data from parsed datasets.
    Maps generic form fields to insurance claim terminology.
    """
    
    # Insurance-specific field mappings
    INSURANCE_FIELD_MAPPING = {
        # FUNSD mappings
        'HEADER': ['claim_id', 'policy_number', 'hospital_name'],
        'QUESTION': ['field_label', 'prompt'],
        'ANSWER': ['field_value', 'response'],
        
        # SROIE-like mappings for hospital bills
        'COMPANY': ['hospital_name', 'clinic_name', 'healthcare_provider'],
        'DATE': ['admission_date', 'discharge_date', 'service_date', 'billing_date'],
        'ADDRESS': ['hospital_address', 'patient_address'],
        'TOTAL': ['total_amount', 'claimed_amount', 'billed_amount'],
    }
    
    # Insurance-specific vocabulary for synthetic augmentation
    INSURANCE_VOCAB = {
        'procedures': [
            'appendectomy', 'cholecystectomy', 'cardiac_catheterization',
            'coronary_angioplasty', 'knee_replacement', 'hip_replacement',
            'cesarean_section', 'hysterectomy', 'cataract_surgery',
            'dialysis', 'chemotherapy', 'radiation_therapy'
        ],
        'diagnoses': [
            'acute_appendicitis', 'cholecystitis', 'myocardial_infarction',
            'pneumonia', 'diabetes_mellitus', 'hypertension',
            'chronic_kidney_disease', 'cancer', 'fracture'
        ],
        'field_labels': [
            'Patient Name', 'Policy Number', 'Claim ID', 'Date of Admission',
            'Date of Discharge', 'Diagnosis', 'Procedure', 'Total Amount',
            'Hospital Name', 'Doctor Name', 'ICD Code', 'Room Charges',
            'Medicine Charges', 'Investigation Charges', 'Consultation Fees'
        ]
    }
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_training_data(
        self, 
        funsd_examples: List[OCRTrainingExample],
        sroie_examples: List[OCRTrainingExample],
        docvqa_examples: List[OCRTrainingExample]
    ) -> Dict[str, Any]:
        """Generate unified training dataset with insurance context."""
        
        training_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_examples': 0,
                'datasets': {
                    'funsd': len(funsd_examples),
                    'sroie': len(sroie_examples),
                    'docvqa': len(docvqa_examples)
                },
                'document_types': ['form', 'receipt', 'document_qa']
            },
            'ner_training': [],      # For NER model training
            'layout_training': [],   # For layout understanding
            'vqa_training': [],      # For document Q&A
            'ocr_training': []       # For OCR fine-tuning
        }
        
        # Process FUNSD for form understanding
        for example in funsd_examples:
            # NER training data
            if example.ner_tags:
                training_data['ner_training'].append({
                    'id': example.id,
                    'tokens': example.words,
                    'ner_tags': example.ner_tags,
                    'bboxes': example.bboxes,
                    'source': 'FUNSD',
                    'application': 'form_understanding'
                })
            
            # Layout training data
            training_data['layout_training'].append({
                'id': example.id,
                'text': example.text,
                'words': example.words,
                'bboxes': example.bboxes,
                'image_url': example.image_url,
                'source': 'FUNSD'
            })
        
        # Process SROIE for receipt/bill OCR
        for example in sroie_examples:
            # Map receipt fields to insurance billing fields
            mapped_tags = self._map_to_insurance_tags(example.ner_tags or [])
            
            training_data['ner_training'].append({
                'id': example.id,
                'tokens': example.words,
                'ner_tags': mapped_tags,
                'bboxes': example.bboxes,
                'source': 'SROIE',
                'application': 'bill_extraction'
            })
            
            # OCR training data
            training_data['ocr_training'].append({
                'id': example.id,
                'text': example.text,
                'words': example.words,
                'bboxes': example.bboxes,
                'source': 'SROIE',
                'document_type': 'receipt_bill'
            })
        
        # Process DocVQA for document Q&A
        for example in docvqa_examples:
            training_data['vqa_training'].append({
                'id': example.id,
                'question': example.question,
                'answer': example.answer,
                'image_url': example.image_url,
                'question_types': example.metadata.get('question_types', []),
                'source': 'DocVQA',
                'application': 'document_qa'
            })
        
        training_data['metadata']['total_examples'] = (
            len(training_data['ner_training']) +
            len(training_data['layout_training']) +
            len(training_data['vqa_training']) +
            len(training_data['ocr_training'])
        )
        
        return training_data
    
    def _map_to_insurance_tags(self, tags: List[str]) -> List[str]:
        """Map generic NER tags to insurance-specific tags."""
        mapping = {
            'O': 'O',
            'B-COMPANY': 'B-HOSPITAL',
            'I-COMPANY': 'I-HOSPITAL',
            'B-DATE': 'B-SERVICE_DATE',
            'I-DATE': 'I-SERVICE_DATE',
            'B-ADDRESS': 'B-FACILITY_ADDRESS',
            'I-ADDRESS': 'I-FACILITY_ADDRESS',
            'B-TOTAL': 'B-AMOUNT',
            'I-TOTAL': 'I-AMOUNT'
        }
        return [mapping.get(tag, tag) for tag in tags]
    
    def save_training_data(self, training_data: Dict[str, Any]) -> Dict[str, str]:
        """Save training data to files."""
        saved_files = {}
        
        # Save main training data
        main_file = self.output_dir / "insurance_ocr_training.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        saved_files['main'] = str(main_file)
        logger.info(f"Saved main training data to {main_file}")
        
        # Save NER training data in CoNLL-like format
        ner_file = self.output_dir / "ner_training.jsonl"
        with open(ner_file, 'w', encoding='utf-8') as f:
            for item in training_data['ner_training']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        saved_files['ner'] = str(ner_file)
        logger.info(f"Saved NER training data to {ner_file}")
        
        # Save VQA training data
        vqa_file = self.output_dir / "vqa_training.jsonl"
        with open(vqa_file, 'w', encoding='utf-8') as f:
            for item in training_data['vqa_training']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        saved_files['vqa'] = str(vqa_file)
        logger.info(f"Saved VQA training data to {vqa_file}")
        
        # Save OCR training data
        ocr_file = self.output_dir / "ocr_training.jsonl"
        with open(ocr_file, 'w', encoding='utf-8') as f:
            for item in training_data['ocr_training']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        saved_files['ocr'] = str(ocr_file)
        logger.info(f"Saved OCR training data to {ocr_file}")
        
        # Save layout training data
        layout_file = self.output_dir / "layout_training.jsonl"
        with open(layout_file, 'w', encoding='utf-8') as f:
            for item in training_data['layout_training']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        saved_files['layout'] = str(layout_file)
        logger.info(f"Saved layout training data to {layout_file}")
        
        # Save metadata summary
        summary_file = self.output_dir / "training_summary.json"
        summary = {
            'created_at': training_data['metadata']['created_at'],
            'total_examples': training_data['metadata']['total_examples'],
            'datasets': training_data['metadata']['datasets'],
            'files_generated': saved_files,
            'training_categories': {
                'ner_training': len(training_data['ner_training']),
                'layout_training': len(training_data['layout_training']),
                'vqa_training': len(training_data['vqa_training']),
                'ocr_training': len(training_data['ocr_training'])
            }
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = str(summary_file)
        logger.info(f"Saved training summary to {summary_file}")
        
        return saved_files


def main():
    """Main entry point for OCR training data processing."""
    # Set up paths
    data_dir = Path(__file__).parent
    
    logger.info("=" * 60)
    logger.info("OCR Training Data Processor")
    logger.info("=" * 60)
    
    # Parse each dataset
    logger.info("\n1. Parsing FUNSD dataset...")
    funsd_parser = FUNSDParser(data_dir)
    funsd_examples = funsd_parser.parse()
    
    logger.info("\n2. Parsing SROIE dataset...")
    sroie_parser = SROIEParser(data_dir)
    sroie_examples = sroie_parser.parse()
    
    logger.info("\n3. Parsing DocVQA dataset...")
    docvqa_parser = DocVQAParser(data_dir)
    docvqa_examples = docvqa_parser.parse()
    
    # Generate unified training data
    logger.info("\n4. Generating unified training data...")
    generator = InsuranceOCRDataGenerator(data_dir / "processed")
    training_data = generator.generate_training_data(
        funsd_examples, sroie_examples, docvqa_examples
    )
    
    # Save training data
    logger.info("\n5. Saving training data...")
    saved_files = generator.save_training_data(training_data)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total examples processed: {training_data['metadata']['total_examples']}")
    logger.info(f"  - FUNSD: {len(funsd_examples)} examples")
    logger.info(f"  - SROIE: {len(sroie_examples)} examples")
    logger.info(f"  - DocVQA: {len(docvqa_examples)} examples")
    logger.info("\nGenerated files:")
    for name, path in saved_files.items():
        logger.info(f"  - {name}: {path}")
    
    return training_data


if __name__ == "__main__":
    main()
