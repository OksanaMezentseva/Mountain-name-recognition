# Named Entity Recognition for Mountain Names

This project focuses on Named Entity Recognition (NER) to identify mountain names in texts. It involves creating a dataset, fine-tuning a BERT-based model, and providing scripts for training, inference, and evaluation.

---

## Dataset
We use a dataset specifically designed for NER of mountain names. The dataset can be downloaded from the Hugging Face Hub:

[Download Dataset](https://huggingface.co/datasets/telord/mountains-ner-dataset?row=16)

Place the dataset in the `data/` folder of the project directory.

---

## Model Weights
The fine-tuned model weights are available for download. Use the following link to download the model weights:

[Download Model Weights](https://drive.google.com/file/d/1iI1r77bcZgGY-6bE524G3MoGTpRMzGP0/view?usp=sharing)

After downloading, extract and place the model weights in the `model/` folder.

---

## Requirements
To install all necessary dependencies, navigate to the project root directory and run:
./Data_Science_Test_Task/Task_1_NER/


```pip install -r requirements.txt```


## Train the Model
To train the model, run:

```python3 src/main.py```
This script will fine-tune the BERT-based model on the NER dataset and save the model and tokenizer in the model/ folder.

## Inference
To perform inference on new text, run:

```python3 src/inference_ner.py```
You can modify the input text in the script or adapt it for your use case.

## Evaluate the Model
To evaluate the model on the test dataset, run:

```python3 src/evaluate_model.py```
This will provide a classification report with precision, recall, and F1-score.

Results
The trained model is capable of identifying mountain names in texts with high accuracy. Example:

Input: "Mount Everest is the highest mountain."
Output: Token predictions: [Mount: B-MOUNTAIN, Everest: I-MOUNTAIN, is: O, the: O, highest: O, mountain: O]
