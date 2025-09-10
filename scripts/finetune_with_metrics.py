#!/usr/bin/env python3
"""
Business Name Generator - Fine-tuning Script with Training Metrics
This script performs LoRA fine-tuning on TinyLlama with comprehensive metrics tracking.
"""

import torch
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_dataset(data, tokenizer, max_length=256):
    """Prepare dataset for training"""
    def tokenize_function(examples):
        texts = []
        for desc, target in zip(examples['description'], examples['target_name']):
            text = f"### Instruction:\\nGenerate a business name for the following description.\\n\\n### Description:\\n{desc}\\n\\n### Business Name:\\n{target}"
            texts.append(text)
        
        return tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    
    dataset_dict = {
        'description': [item['description'] for item in data],
        'target_name': [item['target_name'] for item in data]
    }
    dataset = Dataset.from_dict(dataset_dict)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

class MetricsTracker(Trainer):
    """Custom Trainer class with comprehensive metrics tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': []
        }
    
    def log(self, logs):
        """Override log method to capture training metrics"""
        super().log(logs)
        
        # Save metrics for plotting
        if 'train_loss' in logs:
            self.training_metrics['train_loss'].append(logs['train_loss'])
        if 'eval_loss' in logs:
            self.training_metrics['eval_loss'].append(logs['eval_loss'])
        if 'learning_rate' in logs:
            self.training_metrics['learning_rate'].append(logs['learning_rate'])
        if 'epoch' in logs:
            self.training_metrics['epoch'].append(logs['epoch'])
        if 'step' in logs:
            self.training_metrics['step'].append(logs['step'])

def plot_training_curves(trainer, output_dir):
    """Create and save training visualization plots"""
    if not trainer.training_metrics['train_loss']:
        print("No training metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fine-Tuning Training Metrics', fontsize=16)
    
    # Training loss
    axes[0, 0].plot(trainer.training_metrics['step'], trainer.training_metrics['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Evaluation loss
    if trainer.training_metrics['eval_loss']:
        eval_steps = trainer.training_metrics['step'][:len(trainer.training_metrics['eval_loss'])]
        axes[0, 1].plot(eval_steps, trainer.training_metrics['eval_loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Evaluation Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Evaluation Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Evaluation Loss (Not Available)')
    
    # Learning rate
    if trainer.training_metrics['learning_rate']:
        lr_steps = trainer.training_metrics['step'][:len(trainer.training_metrics['learning_rate'])]
        axes[1, 0].plot(lr_steps, trainer.training_metrics['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate (Not Available)')
    
    # Combined loss plot
    if trainer.training_metrics['eval_loss']:
        min_len = min(len(trainer.training_metrics['train_loss']), len(trainer.training_metrics['eval_loss']))
        axes[1, 1].plot(trainer.training_metrics['step'][:min_len], 
                      trainer.training_metrics['train_loss'][:min_len], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(trainer.training_metrics['step'][:min_len], 
                      trainer.training_metrics['eval_loss'][:min_len], 'r-', label='Eval', linewidth=2)
        axes[1, 1].set_title('Loss Comparison')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].plot(trainer.training_metrics['step'], trainer.training_metrics['train_loss'], 'b-', linewidth=2)
        axes[1, 1].set_title('Training Loss Only')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plots_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plots_path}")
    
    # Save metrics to JSON
    metrics_path = Path(output_dir) / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(trainer.training_metrics, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")
    
    plt.show()
    return plots_path, metrics_path

def main():
    """Main fine-tuning function"""
    parser = argparse.ArgumentParser(description='Fine-tune TinyLlama for business name generation')
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Base model to fine-tune')
    parser.add_argument('--train_file', required=True, help='Training data JSONL file')
    parser.add_argument('--val_file', required=True, help='Validation data JSONL file')
    parser.add_argument('--output_dir', required=True, help='Output directory for model and results')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per device')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank parameter')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--save_plots', action='store_true', help='Save training plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BUSINESS NAME GENERATOR - FINE-TUNING")
    print("=" * 60)
    print(f"Base Model: {args.model}")
    print(f"Training File: {args.train_file}")
    print(f"Validation File: {args.val_file}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"LoRA Rank: {args.lora_rank}")
    print("=" * 60)
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()
    
    # Load and prepare data
    print("Loading training data...")
    train_data = load_jsonl(args.train_file)
    val_data = load_jsonl(args.val_file)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    val_dataset = prepare_dataset(val_data, tokenizer, args.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb/tensorboard logging
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create trainer with metrics tracking
    trainer = MetricsTracker(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning...")
    print("-" * 40)
    
    # Train the model
    trainer.train()
    
    print("-" * 40)
    print("Fine-tuning completed!")
    
    # Create training visualizations
    print("Creating training curves...")
    if args.save_plots or trainer.training_metrics['train_loss']:
        plot_training_curves(trainer, output_dir)
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training summary
    summary = {
        'model': args.model,
        'training_args': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'lora_rank': args.lora_rank,
            'max_length': args.max_length
        },
        'data_stats': {
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        },
        'final_metrics': {
            'final_train_loss': trainer.training_metrics['train_loss'][-1] if trainer.training_metrics['train_loss'] else None,
            'final_eval_loss': trainer.training_metrics['eval_loss'][-1] if trainer.training_metrics['eval_loss'] else None,
            'total_steps': len(trainer.training_metrics['step'])
        }
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    print("=" * 60)
    print("FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
