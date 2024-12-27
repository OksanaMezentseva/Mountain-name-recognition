
from datasets import DatasetDict

def split_dataset(dataset, test_size=0.2):
    """
    Split the dataset into train and test sets.

    Args:
    - dataset (Dataset): Tokenized dataset.
    - test_size (float): Proportion of the test set.

    Returns:
    - DatasetDict: A dictionary with 'train' and 'test' datasets.
    """
    # Split the dataset into train and test subsets
    return dataset.train_test_split(test_size=test_size)
