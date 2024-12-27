from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report
import torch
import os
from data_split import split_dataset
import numpy as np

# Define paths dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
model_path = os.path.join(project_root, "model")
dataset_path = os.path.join(project_root, "data", "mountains_ner.csv")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Label mapping
label_mapping = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}
id_to_label = {v: k for k, v in label_mapping.items()}

def predict_batch(batch):
    """
    Predicts labels for a batch of tokenized inputs.

    Args:
    - batch (dict): A dictionary containing "tokens" and "labels".

    Returns:
    - true_labels (list): The ground truth labels in human-readable format.
    - predicted_labels (list): The predicted labels in human-readable format.
    """
    # Extract tokens and labels from the batch
    tokens = batch["tokens"]  # List of tokens or tokenized sentences
    labels = batch["labels"]

    # Ensure tokens are in the correct format
    if isinstance(tokens, str):
        tokens = [tokens]  # Convert single string to a list for tokenization

    # Tokenize the batch
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,  # Ensures proper handling of word-level tokens
        truncation=True,  # Truncate sequences that exceed the model's max length
        padding=True,  # Pad sequences to the same length
        return_tensors="pt"  # Return PyTorch tensors for model compatibility
    )

    # Perform inference with the model
    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    # Extract predictions from model output
    predictions = torch.argmax(outputs.logits, dim=2).numpy()

    # Convert labels to NumPy array if they are not already
    if isinstance(labels, list):
        labels = np.array(labels)

    # Convert predictions and labels to human-readable format
    predicted_labels = [
        [id_to_label[label] for label in sentence if label != -100]
        for sentence in predictions
    ]
    true_labels = [
        [id_to_label[label] for label in sentence if label != -100]
        for sentence in labels
    ]

    return true_labels, predicted_labels

# Load and split the dataset
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

from data_preprocessing import load_and_convert_labels
from tokenize_and_align_labels import tokenize_and_align_labels
from datasets import Dataset

# Load and preprocess dataset
processed_df = load_and_convert_labels(dataset_path)

# Tokenize and align labels
tokenized_dataset = Dataset.from_pandas(processed_df).map(
    lambda x: tokenize_and_align_labels(x, label_mapping), batched=False
)

# Split the dataset into train and test sets
split_data = split_dataset(tokenized_dataset, test_size=0.2)
test_dataset = split_data["test"]

# Evaluate on the test set
all_true_labels = []
all_predicted_labels = []

for example in test_dataset:
    true_labels, predicted_labels = predict_batch(example)
    all_true_labels.extend(true_labels)
    all_predicted_labels.extend(predicted_labels)

# Calculate metrics
print(classification_report(all_true_labels, all_predicted_labels, target_names=label_mapping.keys()))