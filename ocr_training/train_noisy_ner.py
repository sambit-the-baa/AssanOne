"""
BERT NER Training with Noisy OCR Data
- Generate synthetic OCR errors
- Fine-tune existing BERT NER model on noisy data
- Make model robust to real-world OCR output
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OCRNoiseConfig:
    """Configuration for synthetic OCR noise generation"""
    # Probability of each error type
    char_substitution_prob: float = 0.05
    char_deletion_prob: float = 0.02
    char_insertion_prob: float = 0.02
    word_merge_prob: float = 0.03
    word_split_prob: float = 0.02
    case_change_prob: float = 0.03
    
    # OCR-specific errors
    add_ocr_artifacts: bool = True
    add_line_noise: bool = True


class OCRNoiseGenerator:
    """Generate synthetic OCR-like noise for training data"""
    
    # Common OCR character confusions
    CHAR_CONFUSIONS = {
        'o': ['0', 'O', 'c'],
        'O': ['0', 'o', 'Q'],
        '0': ['O', 'o', 'D'],
        'i': ['1', 'l', '|', 'I'],
        'I': ['1', 'l', '|', 'i'],
        'l': ['1', 'I', '|', 'i'],
        '1': ['l', 'I', '|', 'i'],
        's': ['5', 'S'],
        'S': ['5', 's'],
        '5': ['S', 's'],
        'a': ['@', 'A'],
        'e': ['c', '3'],
        'g': ['9', 'q'],
        '9': ['g', 'q'],
        'n': ['m', 'h', 'ri'],
        'm': ['n', 'rn', 'nn'],
        'h': ['b', 'li', 'n'],
        'd': ['cl', 'b'],
        'w': ['vv', 'uu', 'W'],
        'u': ['v', 'U'],
        'B': ['8', '3', 'b'],
        '8': ['B', '3'],
        'Z': ['2', 'z'],
        '2': ['Z', 'z'],
    }
    
    # OCR artifacts
    ARTIFACTS = ['|', '\\', '/', '_', '~', '`', '^']
    
    def __init__(self, config: OCRNoiseConfig = None):
        self.config = config or OCRNoiseConfig()
    
    def add_noise(self, text: str) -> str:
        """Add synthetic OCR noise to text"""
        chars = list(text)
        result = []
        i = 0
        
        while i < len(chars):
            char = chars[i]
            
            # Character substitution
            if random.random() < self.config.char_substitution_prob:
                if char in self.CHAR_CONFUSIONS:
                    char = random.choice(self.CHAR_CONFUSIONS[char])
            
            # Character deletion
            if random.random() < self.config.char_deletion_prob and len(chars) > 3:
                i += 1
                continue
            
            # Character insertion
            if random.random() < self.config.char_insertion_prob:
                if self.config.add_ocr_artifacts and random.random() < 0.3:
                    result.append(random.choice(self.ARTIFACTS))
                else:
                    result.append(char)  # Duplicate
            
            # Case change
            if random.random() < self.config.case_change_prob:
                char = char.swapcase()
            
            result.append(char)
            i += 1
        
        text = ''.join(result)
        
        # Word-level noise
        words = text.split()
        noisy_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Word merge (join with next word)
            if i < len(words) - 1 and random.random() < self.config.word_merge_prob:
                word = word + words[i + 1]
                i += 1
            
            # Word split (insert space)
            elif len(word) > 4 and random.random() < self.config.word_split_prob:
                split_pos = random.randint(2, len(word) - 2)
                word = word[:split_pos] + ' ' + word[split_pos:]
            
            noisy_words.append(word)
            i += 1
        
        return ' '.join(noisy_words)
    
    def add_noise_to_tokens(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Add noise while preserving token-label alignment.
        Returns new tokens and aligned labels.
        """
        noisy_tokens = []
        noisy_labels = []
        
        for token, label in zip(tokens, labels):
            # Apply noise to token
            noisy_token = self.add_noise(token)
            
            # Handle potential word splits
            split_tokens = noisy_token.split()
            
            if len(split_tokens) > 1:
                # Word was split - first token keeps original label, rest get I- prefix
                noisy_tokens.append(split_tokens[0])
                noisy_labels.append(label)
                
                for split_token in split_tokens[1:]:
                    noisy_tokens.append(split_token)
                    # Convert B- to I- for continuation
                    if label.startswith('B-'):
                        noisy_labels.append('I-' + label[2:])
                    else:
                        noisy_labels.append(label)
            else:
                noisy_tokens.append(noisy_token)
                noisy_labels.append(label)
        
        return noisy_tokens, noisy_labels


@dataclass
class NERExample:
    """Single NER training example"""
    id: str
    tokens: List[str]
    labels: List[str]
    text: str = ""


class NoisyNERDatasetGenerator:
    """Generate noisy NER training data"""
    
    def __init__(self, noise_generator: OCRNoiseGenerator = None):
        self.noise_generator = noise_generator or OCRNoiseGenerator()
    
    def create_synthetic_examples(self, num_examples: int = 1000) -> List[NERExample]:
        """Create synthetic insurance claim NER examples with noise"""
        examples = []
        
        # Template patterns for insurance claims
        templates = self._get_templates()
        
        for i in range(num_examples):
            template = random.choice(templates)
            tokens, labels = self._fill_template(template)
            
            # Add OCR noise
            noisy_tokens, noisy_labels = self.noise_generator.add_noise_to_tokens(tokens, labels)
            
            example = NERExample(
                id=f"synthetic_{i}",
                tokens=noisy_tokens,
                labels=noisy_labels,
                text=' '.join(noisy_tokens)
            )
            examples.append(example)
        
        return examples
    
    def _get_templates(self) -> List[Dict]:
        """Get template patterns for synthetic data"""
        return [
            {
                'pattern': ['Patient', 'Name', ':', '{PATIENT_NAME}'],
                'label_map': {'Patient': 'O', 'Name': 'O', ':': 'O', '{PATIENT_NAME}': 'PATIENT_NAME'}
            },
            {
                'pattern': ['Hospital', ':', '{HOSPITAL}', ',', '{FACILITY_ADDRESS}'],
                'label_map': {'Hospital': 'O', ':': 'O', ',': 'O', '{HOSPITAL}': 'HOSPITAL', '{FACILITY_ADDRESS}': 'FACILITY_ADDRESS'}
            },
            {
                'pattern': ['Policy', 'No', ':', '{POLICY_NUMBER}'],
                'label_map': {'Policy': 'O', 'No': 'O', ':': 'O', '{POLICY_NUMBER}': 'POLICY_NUMBER'}
            },
            {
                'pattern': ['Claim', 'ID', ':', '{CLAIM_ID}'],
                'label_map': {'Claim': 'O', 'ID': 'O', ':': 'O', '{CLAIM_ID}': 'CLAIM_ID'}
            },
            {
                'pattern': ['Admission', 'Date', ':', '{ADMISSION_DATE}'],
                'label_map': {'Admission': 'O', 'Date': 'O', ':': 'O', '{ADMISSION_DATE}': 'ADMISSION_DATE'}
            },
            {
                'pattern': ['Discharge', 'Date', ':', '{DISCHARGE_DATE}'],
                'label_map': {'Discharge': 'O', 'Date': 'O', ':': 'O', '{DISCHARGE_DATE}': 'DISCHARGE_DATE'}
            },
            {
                'pattern': ['Diagnosis', ':', '{DIAGNOSIS}'],
                'label_map': {'Diagnosis': 'O', ':': 'O', '{DIAGNOSIS}': 'DIAGNOSIS'}
            },
            {
                'pattern': ['Total', 'Amount', ':', 'Rs.', '{AMOUNT}'],
                'label_map': {'Total': 'O', 'Amount': 'O', ':': 'O', 'Rs.': 'O', '{AMOUNT}': 'AMOUNT'}
            },
            {
                'pattern': ['Dr.', '{DOCTOR_NAME}', 'treated', 'patient', '{PATIENT_NAME}'],
                'label_map': {'Dr.': 'O', '{DOCTOR_NAME}': 'DOCTOR_NAME', 'treated': 'O', 'patient': 'O', '{PATIENT_NAME}': 'PATIENT_NAME'}
            },
        ]
    
    def _fill_template(self, template: Dict) -> Tuple[List[str], List[str]]:
        """Fill template with random data and return tokens/labels"""
        tokens = []
        labels = []
        
        for pattern_token in template['pattern']:
            label_type = template['label_map'].get(pattern_token, 'O')
            
            if pattern_token.startswith('{') and pattern_token.endswith('}'):
                # Replace placeholder with random data
                entity_type = pattern_token[1:-1]
                entity_tokens = self._generate_entity(entity_type)
                
                for i, token in enumerate(entity_tokens):
                    tokens.append(token)
                    if i == 0:
                        labels.append(f'B-{label_type}')
                    else:
                        labels.append(f'I-{label_type}')
            else:
                tokens.append(pattern_token)
                labels.append(label_type)
        
        return tokens, labels
    
    def _generate_entity(self, entity_type: str) -> List[str]:
        """Generate random entity of given type"""
        if entity_type == 'PATIENT_NAME':
            first_names = ['Sandhya', 'Zoya', 'Rahul', 'Priya', 'Amit', 'Neha', 'Vikram', 'Meera', 'Suresh', 'Kavita']
            middle_names = ['Raghunath', 'Samir', 'Kumar', 'Devi', 'Singh', 'Rani', 'Prasad', 'Lakshmi']
            last_names = ['Kate', 'Shaikh', 'Sharma', 'Patel', 'Gupta', 'Singh', 'Kumar', 'Verma', 'Yadav', 'Reddy']
            
            name = [random.choice(first_names)]
            if random.random() > 0.3:
                name.append(random.choice(middle_names))
            name.append(random.choice(last_names))
            return name
        
        elif entity_type == 'HOSPITAL':
            prefixes = ['City', 'Apollo', 'Max', 'Fortis', 'AIIMS', 'Government', 'District', 'Basil']
            suffixes = ['Hospital', 'Medical Center', 'Healthcare', 'Clinic']
            return [random.choice(prefixes), random.choice(suffixes)]
        
        elif entity_type == 'POLICY_NUMBER':
            prefix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
            number = ''.join(random.choices('0123456789', k=8))
            return [f'{prefix}{number}']
        
        elif entity_type == 'CLAIM_ID':
            return [str(random.randint(1000000, 9999999))]
        
        elif entity_type in ['ADMISSION_DATE', 'DISCHARGE_DATE', 'SERVICE_DATE']:
            day = str(random.randint(1, 28)).zfill(2)
            month = str(random.randint(1, 12)).zfill(2)
            year = str(random.randint(2020, 2024))
            return [f'{day}/{month}/{year}']
        
        elif entity_type == 'DIAGNOSIS':
            diagnoses = [
                ['Fever'], ['Viral', 'Fever'], ['Dengue'], ['Typhoid'],
                ['Fracture'], ['Appendicitis'], ['Diabetes'], ['Hypertension'],
                ['COVID-19'], ['Pneumonia'], ['Cardiac', 'Arrest']
            ]
            return random.choice(diagnoses)
        
        elif entity_type == 'AMOUNT':
            amount = random.randint(1000, 1000000)
            return [f'{amount:,}']
        
        elif entity_type == 'DOCTOR_NAME':
            first = random.choice(['Rajesh', 'Sunita', 'Arun', 'Meena', 'Vijay', 'Anita'])
            last = random.choice(['Kumar', 'Sharma', 'Gupta', 'Singh', 'Patel'])
            return [first, last]
        
        elif entity_type == 'FACILITY_ADDRESS':
            cities = ['Mumbai', 'Delhi', 'Pune', 'Chennai', 'Bangalore', 'Hyderabad']
            return [random.choice(cities)]
        
        elif entity_type == 'ICD_CODE':
            return [f'J{random.randint(10, 99)}.{random.randint(0, 9)}']
        
        elif entity_type == 'PROCEDURE':
            procedures = [
                ['Blood', 'Test'], ['X-Ray'], ['MRI', 'Scan'], ['CT', 'Scan'],
                ['Surgery'], ['Dialysis'], ['Chemotherapy']
            ]
            return random.choice(procedures)
        
        return ['Unknown']
    
    def augment_existing_data(self, examples: List[NERExample], 
                               augmentation_factor: int = 3) -> List[NERExample]:
        """Augment existing clean data with noisy versions"""
        augmented = []
        
        for example in examples:
            # Keep original
            augmented.append(example)
            
            # Create noisy versions
            for i in range(augmentation_factor):
                noisy_tokens, noisy_labels = self.noise_generator.add_noise_to_tokens(
                    example.tokens, example.labels
                )
                
                noisy_example = NERExample(
                    id=f"{example.id}_noisy_{i}",
                    tokens=noisy_tokens,
                    labels=noisy_labels,
                    text=' '.join(noisy_tokens)
                )
                augmented.append(noisy_example)
        
        return augmented


def prepare_training_data(examples: List[NERExample], 
                          label2id: Dict[str, int]) -> List[Dict[str, Any]]:
    """Convert NER examples to training format"""
    training_data = []
    
    for example in examples:
        # Convert labels to IDs
        label_ids = []
        for label in example.labels:
            if label in label2id:
                label_ids.append(label2id[label])
            else:
                label_ids.append(label2id.get('O', 0))
        
        training_data.append({
            'id': example.id,
            'tokens': example.tokens,
            'ner_tags': label_ids
        })
    
    return training_data


def train_noisy_bert_ner(
    model_path: str,
    output_path: str,
    num_synthetic_examples: int = 2000,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Fine-tune existing BERT NER model on noisy OCR data.
    
    Args:
        model_path: Path to existing trained model
        output_path: Path to save fine-tuned model
        num_synthetic_examples: Number of synthetic examples to generate
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    try:
        import torch
        from transformers import (
            BertTokenizerFast, 
            BertForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification
        )
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        raise
    
    logger.info("Loading existing model...")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    
    # Get label mappings from model config
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    logger.info(f"Model has {len(id2label)} labels")
    
    # Generate noisy training data
    logger.info(f"Generating {num_synthetic_examples} synthetic noisy examples...")
    noise_config = OCRNoiseConfig(
        char_substitution_prob=0.08,
        char_deletion_prob=0.03,
        char_insertion_prob=0.03,
        word_merge_prob=0.04,
        word_split_prob=0.03,
    )
    
    generator = NoisyNERDatasetGenerator(OCRNoiseGenerator(noise_config))
    synthetic_examples = generator.create_synthetic_examples(num_synthetic_examples)
    
    # Prepare training data
    training_data = prepare_training_data(synthetic_examples, label2id)
    
    logger.info(f"Prepared {len(training_data)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(training_data)
    
    # Tokenize
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples['tokens'],
            truncation=True,
            padding='max_length',
            max_length=128,
            is_split_into_words=True
        )
        
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx] if word_idx < len(label) else 0)
                else:
                    # Same word - use I- version if B-
                    prev_label = label[word_idx] if word_idx < len(label) else 0
                    label_ids.append(prev_label)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized['labels'] = labels
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    
    # Split dataset
    split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        warmup_ratio=0.1,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Training complete!")
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Train BERT NER on noisy OCR data")
    parser.add_argument("--model-path", default="ocr_training/models/bert_ner",
                        help="Path to existing BERT NER model")
    parser.add_argument("--output-path", default="ocr_training/models/bert_ner_noisy",
                        help="Path to save fine-tuned model")
    parser.add_argument("--num-examples", type=int, default=2000,
                        help="Number of synthetic examples")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    
    args = parser.parse_args()
    
    # Generate sample data first
    print("=" * 60)
    print("Generating sample noisy data...")
    print("=" * 60)
    
    noise_gen = OCRNoiseGenerator()
    data_gen = NoisyNERDatasetGenerator(noise_gen)
    
    # Show sample
    samples = data_gen.create_synthetic_examples(5)
    for sample in samples:
        print(f"\nID: {sample.id}")
        print(f"Text: {sample.text}")
        print(f"Tokens: {sample.tokens}")
        print(f"Labels: {sample.labels}")
    
    print("\n" + "=" * 60)
    print("To train the model, run with --train flag")
    print("=" * 60)
