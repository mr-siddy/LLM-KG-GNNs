"""
example_generator.py

Functions for generating training examples for the recommendation system.
"""

import pandas as pd
import json
from datetime import timedelta
from typing import Dict, List, Any, Optional


def generate_single_example(original_customer_id: str, 
                           customer_id_mapping: Dict[str, int],
                           filtered_customers: pd.DataFrame, 
                           filtered_transactions: pd.DataFrame,
                           filtered_articles: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a rich training example for a single customer.
    
    Args:
        original_customer_id: Original customer ID
        customer_id_mapping: Dictionary mapping original customer IDs to simple numeric IDs
        filtered_customers: DataFrame containing customer data
        filtered_transactions: DataFrame containing transaction data
        filtered_articles: DataFrame containing article data
        
    Returns:
        Dictionary containing the training example
    """
    simple_customer_id = customer_id_mapping.get(original_customer_id)
    if simple_customer_id is None:
        raise ValueError(f"Mapping for customer {original_customer_id} not found")

    # Retrieve Customer Profile
    cust_profile = filtered_customers[filtered_customers['customer_id'] == original_customer_id]
    if cust_profile.empty:
        raise ValueError(f"No customer profile found for customer {original_customer_id}")
    cust_profile_dict = cust_profile.iloc[0].to_dict()

    # Retrieve Transaction History and Merge Article Details
    cust_transactions = filtered_transactions[filtered_transactions['customer_id'] == original_customer_id]
    if not cust_transactions.empty:
        # Merge each transaction with article features (on article_id)
        cust_transactions = cust_transactions.merge(filtered_articles, on='article_id', how='left')
        transactions_list = cust_transactions.to_dict(orient='records')
    else:
        transactions_list = []

    # Construct the Rich Input Data
    input_data = {
        "Simple_Customer_ID": simple_customer_id,
        "Original_Customer_ID": original_customer_id,
        "Customer_Profile": cust_profile_dict,
        "Transaction_History": transactions_list,
        "Instructions": (
            "Based on the above comprehensive customer profile, full transaction history (with detailed article information), "
            "and all available features, generate product recommendations for the next 7 days. "
            "Each recommendation must include the day number, article_id, and product_name in JSON format."
        )
    }
    
    return input_data


def generate_bulk_examples(customer_ids: List[str],
                          customer_id_mapping: Dict[str, int],
                          filtered_customers: pd.DataFrame,
                          filtered_transactions: pd.DataFrame,
                          filtered_articles: pd.DataFrame,
                          min_transactions: int = 10,
                          holdout_days: int = 7,
                          max_transactions: int = 50) -> List[Dict[str, str]]:
    """
    Generate bulk training examples for multiple customers with train/test split.
    
    Args:
        customer_ids: List of customer IDs to process
        customer_id_mapping: Dictionary mapping original customer IDs to simple numeric IDs
        filtered_customers: DataFrame containing customer data
        filtered_transactions: DataFrame containing transaction data with 't_dat' as datetime
        filtered_articles: DataFrame containing article data
        min_transactions: Minimum transactions to use fixed holdout period
        holdout_days: Number of days to hold out for testing if sufficient transactions
        max_transactions: Maximum number of historical transactions to include
        
    Returns:
        List of dictionaries containing input/output pairs for training
    """
    # Ensure transactions have datetime format
    if not pd.api.types.is_datetime64_dtype(filtered_transactions['t_dat']):
        filtered_transactions = filtered_transactions.copy()
        filtered_transactions['t_dat'] = pd.to_datetime(filtered_transactions['t_dat'])
    
    # Define relevant columns for each dataset to reduce data size
    relevant_customer_cols = ['customer_id', 'club_member_status', 'fashion_news_frequency', 'age']
    relevant_tx_cols = ['t_dat', 'customer_id', 'article_id', 'price']
    relevant_article_cols = ['article_id', 'prod_name', 'product_type_name', 
                             'product_group_name', 'colour_group_name', 'garment_group_name']

    # Filter columns accordingly
    filtered_customers = filtered_customers[relevant_customer_cols]
    filtered_transactions = filtered_transactions[relevant_tx_cols]
    filtered_articles = filtered_articles[relevant_article_cols]
    
    training_examples = []

    for cust_id in customer_ids:
        # Map the long customer ID to a simple numeric ID
        simple_id = customer_id_mapping.get(cust_id)
        if simple_id is None:
            continue
        
        # Retrieve customer profile and keep only relevant columns, then remove the original customer_id key
        cust_profile_df = filtered_customers[filtered_customers['customer_id'] == cust_id]
        if cust_profile_df.empty:
            continue
        cust_profile = cust_profile_df.iloc[0].to_dict()
        cust_profile.pop('customer_id', None)
        
        # Retrieve the customer's transaction history and sort by date
        cust_tx = filtered_transactions[filtered_transactions['customer_id'] == cust_id].copy()
        cust_tx.sort_values(by='t_dat', inplace=True)
        num_tx = len(cust_tx)
        
        # Apply cutoff strategy based on number of transactions
        if num_tx >= min_transactions:
            last_date = cust_tx['t_dat'].max()
            cutoff_date = last_date - timedelta(days=holdout_days)
            hist_tx = cust_tx[cust_tx['t_dat'] <= cutoff_date]
            future_tx = cust_tx[cust_tx['t_dat'] > cutoff_date]
            strategy_used = f"Fixed {holdout_days}-day cutoff."
        else:
            split_idx = int(num_tx * 0.7)
            hist_tx = cust_tx.iloc[:split_idx]
            future_tx = cust_tx.iloc[split_idx:]
            strategy_used = "70/30 split due to low transaction count."
        
        # Truncate historical transactions if there are too many
        if len(hist_tx) > max_transactions:
            hist_tx = hist_tx.iloc[-max_transactions:]
        
        # Merge historical transactions with article details and drop the original customer_id
        if not hist_tx.empty:
            hist_tx = hist_tx.merge(filtered_articles, on='article_id', how='left')
            hist_tx = hist_tx.drop(columns=["customer_id"], errors='ignore')
            hist_list = hist_tx.to_dict(orient='records')
        else:
            hist_list = []
        
        # Merge future transactions with article details for ground truth and drop the original customer_id
        if not future_tx.empty:
            future_tx = future_tx.merge(filtered_articles, on='article_id', how='left')
            future_tx = future_tx.drop(columns=["customer_id"], errors='ignore')
            future_list = future_tx.to_dict(orient='records')
        else:
            future_list = []
        
        # Construct the rich input using only the simple customer ID and filtered information
        input_data = {
            "Simple_Customer_ID": simple_id,
            "Customer_Profile": cust_profile,
            "Historical_Transactions": hist_list,
            "Instructions": (
                "Based on the above customer profile and historical transaction data (with essential product details), "
                "generate product recommendations for the next 7 days. The output should be in JSON format with one recommendation per day, "
                "including keys: day, article_id, and product_name. "
                f"(Cutoff strategy used: {strategy_used})"
            )
        }
        
        output_data = {
            "Ground_Truth_Future_Transactions": future_list
        }
        
        training_examples.append({
            "input": json.dumps(input_data, indent=2, default=str),
            "output": json.dumps(output_data, indent=2, default=str)
        })
    
    return training_examples


def save_training_examples(examples: List[Dict[str, str]], output_path: str) -> None:
    """
    Save training examples to a JSON file.
    
    Args:
        examples: List of training examples (input/output pairs)
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2, default=str)
    
    print(f"Created training examples for {len(examples)} customers. Saved to {output_path}.")


if __name__ == "__main__":
    # Example usage
    import id_mapper
    
    # Load filtered data
    filtered_customers = pd.read_csv('filtered_customers.csv')
    filtered_transactions = pd.read_csv('filtered_transactions_train.csv')
    filtered_transactions['t_dat'] = pd.to_datetime(filtered_transactions['t_dat'])
    filtered_articles = pd.read_csv('filtered_articles.csv')
    
    # Create or load customer ID mapping
    mapping = id_mapper.create_customer_id_mapping(filtered_customers)
    
    # Get a list of unique customer IDs (first 5 for testing)
    customer_ids = filtered_transactions['customer_id'].unique()[:5]
    
    # Generate examples
    examples = generate_bulk_examples(
        customer_ids,
        mapping,
        filtered_customers,
        filtered_transactions,
        filtered_articles
    )
    
    # Save examples
    save_training_examples(examples, 'test_training_examples.json')
