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

# Example sentence for inference
example_text = "Mount Kilimanjaro is located in Africa."

# Tokenize the input
inputs = tokenizer(
    example_text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Decode predictions
predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
labels = [model.config.id2label[label] for label in predictions]

# Print results
print("Token predictions:")
for token, label in zip(tokens, labels):
    print(f"{token}: {label}")