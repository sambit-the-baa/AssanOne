"""
Auto-generated Training Script
==============================
Model: bert_ner
Task: insurance_ner
Generated: 2025-12-31T09:04:33.737093
"""

import json
import os
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

# Set base directory
BASE_DIR = Path(__file__).parent.resolve()

# Configuration
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = str(BASE_DIR / "models" / "bert_ner")
DATA_DIR = str(BASE_DIR / "processed" / "ner_training.jsonl")
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-05
MAX_SEQ_LENGTH = 512

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Label mapping
LABELS = ['O', 'B-HOSPITAL', 'I-HOSPITAL', 'B-PATIENT_NAME', 'I-PATIENT_NAME', 'B-CLAIM_ID', 'I-CLAIM_ID', 'B-POLICY_NUMBER', 'I-POLICY_NUMBER', 'B-ADMISSION_DATE', 'I-ADMISSION_DATE', 'B-DISCHARGE_DATE', 'I-DISCHARGE_DATE', 'B-DIAGNOSIS', 'I-DIAGNOSIS', 'B-PROCEDURE', 'I-PROCEDURE', 'B-ICD_CODE', 'I-ICD_CODE', 'B-AMOUNT', 'I-AMOUNT', 'B-SERVICE_DATE', 'I-SERVICE_DATE', 'B-FACILITY_ADDRESS', 'I-FACILITY_ADDRESS', 'B-DOCTOR_NAME', 'I-DOCTOR_NAME']
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

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
    
    # Move model to GPU if available
    model = model.to(device)
    
    print("Loading training data...")
    train_data = load_data(DATA_DIR)
    
    # Split data
    split_idx = int(len(train_data) * 0.9)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    # Convert to Dataset format
    train_dataset = Dataset.from_list(train_split)
    val_dataset = Dataset.from_list(val_split)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
    
    # Training arguments with GPU support
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
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
        fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
        dataloader_num_workers=0,  # Windows compatibility
        no_cuda=not torch.cuda.is_available(),  # Force GPU if available
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
    # Start fresh training (GPU will be much faster)
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
