"""
main.py

Main script to run the complete recommendation system pipeline:
1. Load and filter data
2. Create customer ID mappings
3. Generate training examples
4. Fine-tune the language model
"""

import os
import argparse
import pandas as pd
from transformers import AutoTokenizer

# Import our modules
from data_loader import load_and_filter_data, save_filtered_data
from id_mapper import create_customer_id_mapping, save_customer_id_mapping
from example_generator import generate_bulk_examples, save_training_examples
from model_trainer import prepare_dataset, tokenize_dataset, fine_tune_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the recommendation system pipeline')
    
    parser.add_argument('--data_dir', type=str, default='./', 
                        help='Directory containing the input data files')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save output files')
    parser.add_argument('--train_json', type=str, default='product_recommendation_train.json',
                        help='Training JSON file name')
    parser.add_argument('--val_json', type=str, default='product_recommendation_val.json',
                        help='Validation JSON file name')
    parser.add_argument('--transactions_csv', type=str, default='transactions_train.csv',
                        help='Transactions CSV file name')
    parser.add_argument('--customers_csv', type=str, default='customers.csv',
                        help='Customers CSV file name')
    parser.add_argument('--articles_csv', type=str, default='articles.csv',
                        help='Articles CSV file name')
    parser.add_argument('--num_examples', type=int, default=1000,
                        help='Number of training examples to generate')
    parser.add_argument('--model_name', type=str, default='EleutherAI/gpt-neo-1.3B',
                        help='Pre-trained model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--skip_fine_tuning', action='store_true',
                        help='Skip model fine-tuning step')
    
    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and filter data
    print("\n=== Step 1: Loading and filtering data ===")
    train_json_path = os.path.join(args.data_dir, args.train_json)
    val_json_path = os.path.join(args.data_dir, args.val_json)
    transactions_path = os.path.join(args.data_dir, args.transactions_csv)
    customers_path = os.path.join(args.data_dir, args.customers_csv)
    articles_path = os.path.join(args.data_dir, args.articles_csv)
    
    filtered_transactions, filtered_customers, filtered_articles = load_and_filter_data(
        train_json_path, val_json_path, transactions_path, customers_path, articles_path
    )
    
    save_filtered_data(filtered_transactions, filtered_customers, filtered_articles, args.output_dir)
    
    # Step 2: Create customer ID mappings
    print("\n=== Step 2: Creating customer ID mappings ===")
    customer_id_mapping = create_customer_id_mapping(filtered_customers)
    mapping_path = os.path.join(args.output_dir, 'customer_id_mapping.csv')
    save_customer_id_mapping(customer_id_mapping, mapping_path)
    
    # Step 3: Generate training examples
    print(f"\n=== Step 3: Generating {args.num_examples} training examples ===")
    filtered_transactions['t_dat'] = pd.to_datetime(filtered_transactions['t_dat'])
    
    # Get a list of unique customer IDs to process
    customer_ids = filtered_transactions['customer_id'].unique()[:args.num_examples]
    
    training_examples = generate_bulk_examples(
        customer_ids,
        customer_id_mapping,
        filtered_customers,
        filtered_transactions,
        filtered_articles
    )
    
    examples_path = os.path.join(args.output_dir, f'finetuning_training_data_{len(training_examples)}_customers.json')
    save_training_examples(training_examples, examples_path)
    
    # Step 4: Fine-tune the model (optional)
    if not args.skip_fine_tuning:
        print("\n=== Step 4: Fine-tuning the language model ===")
        model_output_dir = os.path.join(args.output_dir, 'finetuned-model')
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Prepare and tokenize dataset
        dataset = prepare_dataset(examples_path)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        # Set training arguments
        training_args = {
            "num_train_epochs": args.epochs,
            "logging_steps": 50,
            "save_total_limit": 2,
            "fp16": True
        }
        
        # Fine-tune model
        fine_tune_model(
            args.model_name,
            tokenized_dataset,
            tokenizer,
            model_output_dir,
            training_args
        )
        
        print(f"\nFine-tuned model saved to {model_output_dir}")
    else:
        print("\n=== Step 4: Skipping model fine-tuning ===")
    
    print("\n=== Pipeline completed successfully! ===")


if __name__ == "__main__":
    main()
