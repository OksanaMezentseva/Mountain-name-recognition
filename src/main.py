# Import necessary modules and functions
from data_preprocessing import load_and_convert_labels
from tokenize_and_align_labels import tokenize_and_align_labels
from data_split import split_dataset
from train_model import train_model
from transformers import AutoTokenizer
from datasets import Dataset
import os

# Define the main function
def main():
    """
    Main function to prepare data, train the NER model, and save results.
    """
    # Define the root directory of the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Define the relative path to the dataset
    dataset_path = os.path.join(project_root, "data", "mountains_ner.csv")

    # Ensure dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    processed_df = load_and_convert_labels(dataset_path)

    # Define label mapping
    label_mapping = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = Dataset.from_pandas(processed_df).map(
        lambda x: tokenize_and_align_labels(x, label_mapping),
        batched=False
    )

    # Split dataset into train and test
    print("Splitting dataset into train and test sets...")
    split_data = split_dataset(tokenized_dataset)
    train_dataset = split_data['train']
    test_dataset = split_data['test']

    # # Limit the size of datasets for testing purposes
    # train_dataset = train_dataset.select(range(1000))  # Use only the first 1000 examples for training
    # test_dataset = test_dataset.select(range(100))    # Use only the first 100 examples for testing

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Train the model
    print("Training the model...")
    trainer = train_model(train_dataset, test_dataset, tokenizer, label_mapping)

    print("Training completed.")

# Run the main function
if __name__ == "__main__":
    main()