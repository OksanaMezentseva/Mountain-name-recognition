import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Define paths dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
model_dir = os.path.join(project_root, "model")

# Load the trained model and tokenizer
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found at {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

def predict(text):
    """
    Perform NER prediction on the given text.
    
    Parameters:
    - text (str): The input text for NER.
    
    Returns:
    - List of tuples with tokens and their predicted labels.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        is_split_into_words=False
    )
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Map predictions back to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    labels = [model.config.id2label[label] for label in predictions]
    
    # Combine tokens with their corresponding labels
    result = list(zip(tokens, labels))
    
    # Filter out special tokens and subwords
    filtered_result = []
    for token, label in result:
        if token not in tokenizer.all_special_tokens and not token.startswith("##"):
            filtered_result.append((token, label))

    return filtered_result

# Example usage
if __name__ == "__main__":
    text = "Mount Everest is the highest mountain."
    print(predict(text))
