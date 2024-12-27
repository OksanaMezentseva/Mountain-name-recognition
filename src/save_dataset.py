from datasets import load_dataset

def load_and_save_dataset():
   
    dataset = load_dataset("telord/mountains-ner-dataset")

    
    dataset['train'].to_csv("Task_1_NER/data/mountains_ner.csv", index=False)
    print("Dataset saved locally as Task_1_NER/data/mountains_ner.csv")

if __name__ == "__main__":
    load_and_save_dataset()