import pandas as pd
import ast
import re  # Import the regex module

# Dictionary to convert numeric labels to NER tags
label_mapping = {
    0: "O",           # Outside any entity
    1: "B-MOUNTAIN",  # Beginning of a mountain name
    2: "I-MOUNTAIN"   # Inside a mountain name
}

def load_and_convert_labels(dataset_path):
    """
    Load the dataset and convert numeric labels to NER tags.

    Parameters:
    dataset_path (str): Path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: DataFrame with converted NER tags.
    """
    # Load the dataset
    dataset_df = pd.read_csv(dataset_path)
    
    # Convert labels column to a proper list format
    dataset_df['labels'] = dataset_df['labels'].apply(
        lambda label_list: re.sub(r'\s+', ', ', label_list)  # Replace all spaces with commas
    )
    
    # Convert numeric labels to text tags using label_mapping
    dataset_df['labels'] = dataset_df['labels'].apply(
        lambda label_list: [label_mapping[label] for label in ast.literal_eval(label_list)]
    )
    
    return dataset_df