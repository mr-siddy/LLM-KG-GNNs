"""
model_trainer.py

Functions for fine-tuning the language model on recommendation data.
"""

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from typing import Dict, Any, Optional


def prepare_dataset(data_path: str) -> Dataset:
    """
    Load and prepare dataset for fine-tuning.
    
    Args:
        data_path: Path to the training data JSON file
    
    Returns:
        Dataset prepared for training
    """
    # Load the JSON file with input and output fields
    dataset = load_dataset("json", data_files={"train": data_path})
    
    # Combine "input" and "output" into a single "text" field for training
    def combine_fields(example):
        example["text"] = example["input"] + "\n\nOutput:\n" + example["output"]
        return example

    prepared_dataset = dataset["train"].map(combine_fields)
    return prepared_dataset


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
    """
    Tokenize the dataset with the specified tokenizer.
    
    Args:
        dataset: Dataset with "text" field
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tokenized dataset
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset and set labels equal to input_ids with padding
    def tokenize_function(example):
        tokenized = tokenizer(
            example["text"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"  # ensures all sequences have the same length
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def fine_tune_model(model_name: str, 
                   tokenized_dataset: Dataset, 
                   tokenizer,
                   output_dir: str = "./model-finetuned-recs",
                   training_args: Optional[Dict[str, Any]] = None) -> None:
    """
    Fine-tune a language model on the recommendation dataset.
    
    Args:
        model_name: Name or path of the pre-trained model
        tokenized_dataset: Tokenized dataset for training
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the fine-tuned model
        training_args: Optional dictionary of training arguments to override defaults
    """
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set up training arguments with defaults
    default_args = {
        "output_dir": output_dir,
        "evaluation_strategy": "no",  # disable evaluation for initial testing
        "logging_steps": 100,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 2,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "fp16": True,
    }
    
    # Override with provided arguments if any
    if training_args:
        default_args.update(training_args)
    
    # Create training arguments
    args = TrainingArguments(**default_args)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start fine-tuning
    print(f"Starting fine-tuning with {model_name}")
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    model_name = "EleutherAI/gpt-neo-1.3B"
    data_path = "finetuning_training_data_1000_customers.json"
    output_dir = "./gptneo-finetuned-recs"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare and tokenize dataset
    dataset = prepare_dataset(data_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Fine-tune model (reduce epochs for demonstration)
    fine_tune_model(
        model_name,
        tokenized_dataset,
        tokenizer,
        output_dir,
        {"num_train_epochs": 1}  # Override for faster testing
    )
