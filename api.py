from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
import glob

MODEL_DIR = "models/bert_sentiment"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    pred = torch.argmax(output.logits, dim=1).item()
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return {"label": label_map[pred]}

@app.get("/metadata")
def get_metadata():
    latest = sorted(glob.glob("metadata/*.json"))[-1]
    with open(latest, "r") as f:
        return json.load(f)

@app.get("/")
def root():
    return {"message": "BERT Sentiment MCP API running."}
