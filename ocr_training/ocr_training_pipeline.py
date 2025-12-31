"""
OCR Training Pipeline
=====================
Training pipeline for fine-tuning OCR/Document AI models 
using the processed insurance document training data.

Supports training:
1. LayoutLM/LayoutLMv2/LayoutLMv3 for document understanding
2. TrOCR for OCR text recognition
3. Donut for document parsing
4. Custom NER models for insurance field extraction

Usage:
    python ocr_training_pipeline.py --model layoutlm --epochs 10
    python ocr_training_pipeline.py --model trocr --task ocr
    python ocr_training_pipeline.py --model ner --task insurance_ner
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    model_name: str
    task: str
    train_file: str
    val_file: Optional[str] = None
    output_dir: str = "models"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_seq_length: int = 512
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42


class OCRTrainingPipeline:
    """Main training pipeline for OCR models."""
    
    # Supported models and their configurations
    SUPPORTED_MODELS = {
        'layoutlm': {
            'hf_name': 'microsoft/layoutlm-base-uncased',
            'task_type': 'token_classification',
            'description': 'LayoutLM for document understanding with layout features'
        },
        'layoutlmv2': {
            'hf_name': 'microsoft/layoutlmv2-base-uncased',
            'task_type': 'token_classification',
            'description': 'LayoutLMv2 with visual features'
        },
        'layoutlmv3': {
            'hf_name': 'microsoft/layoutlmv3-base',
            'task_type': 'token_classification',
            'description': 'LayoutLMv3 unified multimodal model'
        },
        'trocr': {
            'hf_name': 'microsoft/trocr-base-printed',
            'task_type': 'image_to_text',
            'description': 'TrOCR for OCR text recognition'
        },
        'donut': {
            'hf_name': 'naver-clova-ix/donut-base',
            'task_type': 'document_parsing',
            'description': 'Donut for end-to-end document parsing'
        },
        'bert_ner': {
            'hf_name': 'bert-base-uncased',
            'task_type': 'token_classification',
            'description': 'BERT for NER without layout features'
        }
    }
    
    # Insurance-specific label set
    INSURANCE_LABELS = [
        'O',
        'B-HOSPITAL', 'I-HOSPITAL',
        'B-PATIENT_NAME', 'I-PATIENT_NAME',
        'B-CLAIM_ID', 'I-CLAIM_ID',
        'B-POLICY_NUMBER', 'I-POLICY_NUMBER',
        'B-ADMISSION_DATE', 'I-ADMISSION_DATE',
        'B-DISCHARGE_DATE', 'I-DISCHARGE_DATE',
        'B-DIAGNOSIS', 'I-DIAGNOSIS',
        'B-PROCEDURE', 'I-PROCEDURE',
        'B-ICD_CODE', 'I-ICD_CODE',
        'B-AMOUNT', 'I-AMOUNT',
        'B-SERVICE_DATE', 'I-SERVICE_DATE',
        'B-FACILITY_ADDRESS', 'I-FACILITY_ADDRESS',
        'B-DOCTOR_NAME', 'I-DOCTOR_NAME'
    ]
    
    def __init__(self, config: TrainingConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"
        self.models_dir = data_dir / config.output_dir
        self.models_dir.mkdir(exist_ok=True)
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are installed."""
        required_packages = [
            ('torch', 'pytorch'),
            ('transformers', 'transformers'),
            ('datasets', 'datasets'),
        ]
        
        missing = []
        for package, pip_name in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(pip_name)
        
        if missing:
            logger.warning(f"Missing packages: {missing}")
            logger.info("Install with: pip install " + " ".join(missing))
    
    def load_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load training data based on task type."""
        train_data = []
        val_data = []
        
        if self.config.task in ['ner', 'insurance_ner', 'token_classification']:
            # Load NER training data
            ner_file = self.processed_dir / "ner_training.jsonl"
            if ner_file.exists():
                with open(ner_file, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                    # Split 90/10 for train/val
                    split_idx = int(len(data) * 0.9)
                    train_data = data[:split_idx]
                    val_data = data[split_idx:]
                    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val NER examples")
        
        elif self.config.task in ['layout', 'document_understanding']:
            # Load layout training data
            layout_file = self.processed_dir / "layout_training.jsonl"
            if layout_file.exists():
                with open(layout_file, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                    split_idx = int(len(data) * 0.9)
                    train_data = data[:split_idx]
                    val_data = data[split_idx:]
                    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val layout examples")
        
        elif self.config.task in ['vqa', 'document_qa']:
            # Load VQA training data
            vqa_file = self.processed_dir / "vqa_training.jsonl"
            if vqa_file.exists():
                with open(vqa_file, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                    split_idx = int(len(data) * 0.9)
                    train_data = data[:split_idx]
                    val_data = data[split_idx:]
                    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val VQA examples")
        
        elif self.config.task in ['ocr', 'text_recognition']:
            # Load OCR training data
            ocr_file = self.processed_dir / "ocr_training.jsonl"
            if ocr_file.exists():
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                    split_idx = int(len(data) * 0.9)
                    train_data = data[:split_idx]
                    val_data = data[split_idx:]
                    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val OCR examples")
        
        return train_data, val_data
    
    def create_label_mapping(self, train_data: List[Dict]) -> Dict[str, int]:
        """Create label to ID mapping from training data."""
        if self.config.task in ['ner', 'insurance_ner', 'token_classification']:
            # Use insurance-specific labels
            label2id = {label: idx for idx, label in enumerate(self.INSURANCE_LABELS)}
            id2label = {idx: label for label, idx in label2id.items()}
            return label2id, id2label
        
        return {}, {}
    
    def generate_training_script(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        """Generate a training script based on configuration."""
        
        model_info = self.SUPPORTED_MODELS.get(self.config.model_name, {})
        hf_model = model_info.get('hf_name', 'bert-base-uncased')
        
        script = f'''"""
Auto-generated Training Script
==============================
Model: {self.config.model_name}
Task: {self.config.task}
Generated: {datetime.now().isoformat()}
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset

# Configuration
MODEL_NAME = "{hf_model}"
OUTPUT_DIR = "{self.models_dir / self.config.model_name}"
EPOCHS = {self.config.epochs}
BATCH_SIZE = {self.config.batch_size}
LEARNING_RATE = {self.config.learning_rate}
MAX_SEQ_LENGTH = {self.config.max_seq_length}

# Label mapping
LABELS = {self.INSURANCE_LABELS}
label2id = {{label: idx for idx, label in enumerate(LABELS)}}
id2label = {{idx: label for label, idx in label2id.items()}}

def load_data(filepath):
    """Load JSONL data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize inputs and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Convert label to ID
                tag = label[word_idx] if word_idx < len(label) else 'O'
                label_ids.append(label2id.get(tag, label2id['O']))
            else:
                # Same word as previous token - use I- tag if applicable
                tag = label[word_idx] if word_idx < len(label) else 'O'
                label_ids.append(label2id.get(tag, label2id['O']))
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    """Main training function."""
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    print("Loading training data...")
    train_data = load_data("{self.processed_dir / 'ner_training.jsonl'}")
    
    # Split data
    split_idx = int(len(train_data) * 0.9)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    # Convert to Dataset format
    train_dataset = Dataset.from_list(train_split)
    val_dataset = Dataset.from_list(val_split)
    
    print(f"Train samples: {{len(train_dataset)}}, Val samples: {{len(val_dataset)}}")
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! Model saved to {{OUTPUT_DIR}}")

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def run_training(self):
        """Main training entry point."""
        logger.info("=" * 60)
        logger.info("OCR Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Task: {self.config.task}")
        logger.info(f"Epochs: {self.config.epochs}")
        
        # Load training data
        logger.info("\n1. Loading training data...")
        train_data, val_data = self.load_training_data()
        
        if not train_data:
            logger.error("No training data found! Run ocr_training_processor.py first.")
            return False
        
        # Create label mapping
        logger.info("\n2. Creating label mapping...")
        label2id, id2label = self.create_label_mapping(train_data)
        logger.info(f"Labels: {len(label2id)} classes")
        
        # Generate training script
        logger.info("\n3. Generating training script...")
        script = self.generate_training_script(train_data, val_data)
        
        # Save training script
        script_path = self.data_dir / f"train_{self.config.model_name}_{self.config.task}.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        logger.info(f"Saved training script to: {script_path}")
        
        # Save training config
        config_path = self.models_dir / f"{self.config.model_name}_config.json"
        config_dict = {
            'model_name': self.config.model_name,
            'hf_model': self.SUPPORTED_MODELS.get(self.config.model_name, {}).get('hf_name'),
            'task': self.config.task,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'labels': self.INSURANCE_LABELS,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'created_at': datetime.now().isoformat()
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved config to: {config_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SETUP COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nTo start training, run:")
        logger.info(f"  python {script_path}")
        logger.info(f"\nOr install dependencies first:")
        logger.info(f"  pip install torch transformers datasets")
        
        return True


class InsuranceOCRIntegration:
    """
    Integration module for connecting trained OCR models 
    with the fraud detection pipeline.
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.label_map = None
    
    def load_model(self, model_name: str = "bert_ner"):
        """Load trained model for inference."""
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            # Load config for label mapping
            config_path = self.model_dir / f"{model_name}_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.label_map = {i: label for i, label in enumerate(config.get('labels', []))}
            
            logger.info(f"Loaded model from {model_path}")
            return True
            
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            return False
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract insurance-relevant entities from text."""
        if self.model is None:
            logger.warning("Model not loaded. Call load_model() first.")
            return {}
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = outputs.logits.argmax(dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Extract entities
        entities = {
            'hospital': [],
            'patient_name': [],
            'claim_id': [],
            'policy_number': [],
            'diagnosis': [],
            'procedure': [],
            'amount': [],
            'date': []
        }
        
        current_entity = None
        current_tokens = []
        
        for token, pred_id in zip(tokens, predictions[0].tolist()):
            label = self.label_map.get(pred_id, 'O')
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    entity_type = current_entity.lower().replace('b-', '').replace('i-', '')
                    entity_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                    if entity_type in entities:
                        entities[entity_type].append(entity_text.strip())
                
                # Start new entity
                current_entity = label
                current_tokens = [token]
            
            elif label.startswith('I-') and current_entity:
                current_tokens.append(token)
            
            else:
                # Save previous entity
                if current_entity and current_tokens:
                    entity_type = current_entity.lower().replace('b-', '').replace('i-', '')
                    entity_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                    if entity_type in entities:
                        entities[entity_type].append(entity_text.strip())
                
                current_entity = None
                current_tokens = []
        
        return entities
    
    def enhance_extraction(self, tesseract_text: str, existing_data: Dict) -> Dict:
        """
        Enhance existing extraction with ML-based entity recognition.
        
        Args:
            tesseract_text: Raw OCR text from Tesseract
            existing_data: Existing extracted data from rule-based extraction
        
        Returns:
            Enhanced data with ML-extracted entities
        """
        # Extract entities using trained model
        ml_entities = self.extract_entities(tesseract_text)
        
        # Merge with existing data (ML fills gaps, doesn't override good data)
        enhanced = existing_data.copy()
        
        # Only use ML entities if existing data is missing or looks like gibberish
        for field, ml_values in ml_entities.items():
            if not ml_values:
                continue
            
            existing_value = enhanced.get(field, '')
            
            # Check if existing value looks valid
            if not existing_value or self._is_gibberish(existing_value):
                enhanced[field] = ml_values[0] if ml_values else existing_value
        
        return enhanced
    
    def _is_gibberish(self, text: str) -> bool:
        """Check if text appears to be gibberish/OCR noise."""
        if not text or len(text) < 2:
            return True
        
        # Check for too many special characters
        special_count = sum(1 for c in text if not c.isalnum() and c != ' ')
        if special_count / len(text) > 0.4:
            return True
        
        # Check for no vowels (likely gibberish)
        vowels = set('aeiouAEIOU')
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and not any(c in vowels for c in alpha_chars):
            return True
        
        return False


def main():
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description='OCR Training Pipeline')
    parser.add_argument('--model', type=str, default='bert_ner',
                       choices=['layoutlm', 'layoutlmv2', 'layoutlmv3', 'trocr', 'donut', 'bert_ner'],
                       help='Model to train')
    parser.add_argument('--task', type=str, default='insurance_ner',
                       choices=['ner', 'insurance_ner', 'layout', 'vqa', 'ocr'],
                       help='Training task')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model,
        task=args.task,
        train_file="ner_training.jsonl",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    # Set up paths
    data_dir = Path(__file__).parent
    
    # Run pipeline
    pipeline = OCRTrainingPipeline(config, data_dir)
    success = pipeline.run_training()
    
    if success:
        logger.info("\nPipeline completed successfully!")
    else:
        logger.error("\nPipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
