from transformers import AutoTokenizer

# Load the tokenizer for the model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(example, label_mapping):
    """
    Tokenizes sentences and aligns NER labels with tokenized tokens for a single example.
    
    Parameters:
    - example (dict): A single example containing 'tokens' and 'labels'.
    - label_mapping (dict): A dictionary mapping label strings to numeric values.
    
    Returns:
    - dict: Tokenized inputs with aligned labels.
    """
    # Ensure tokens are provided as a list of strings
    tokens = example["tokens"]
    if isinstance(tokens[0], list):
        tokens = [str(token) for sublist in tokens for token in sublist]
    else:
        tokens = [str(token) for token in tokens]

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True
    )
    
    # Prepare labels to align with tokenized inputs
    labels = []
    word_ids = tokenized_inputs.word_ids()  # Map token IDs to word IDs
    previous_word_id = None  # Keep track of the previous word ID for alignment

    for word_id in word_ids:
        if word_id is None:  # Special tokens like [CLS], [SEP]
            labels.append(-100)  # Ignore these tokens in loss computation
        elif word_id != previous_word_id:  # First subtoken of a word
            if word_id < len(example["labels"]):  # Ensure word ID is within bounds
                labels.append(label_mapping[example["labels"][word_id]])  # Map label
            else:
                labels.append(-100)  # Ignore out-of-bounds word IDs
        else:  # Subtokens of a word
            labels.append(-100)  # Ignore subtokens for loss computation
        previous_word_id = word_id  # Update the previous word ID

    # Ensure that the length of labels matches input IDs
    if len(labels) != len(tokenized_inputs["input_ids"]):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(tokenized_inputs['input_ids'])} tokens"
        )

    # Add the aligned labels to the tokenized inputs
    tokenized_inputs["labels"] = labels
    return tokenized_inputs