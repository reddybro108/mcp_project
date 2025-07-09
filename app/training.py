import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from mcp_logger import log_model_metadata

MODEL_DIR = "models/bert_sentiment"

def train_model():
    df = pd.read_csv("data/dataset.csv")
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['label'].map(label_map)
    
    dataset = Dataset.from_pandas(df)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="no",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_dir="logs",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    log_model_metadata(model_dir=MODEL_DIR, metrics={"accuracy": "N/A (demo)"})

if __name__ == "__main__":
    train_model()
