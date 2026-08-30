"""Railway backend: BERT + Logistic Regression"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re, string, json
from pathlib import Path
import numpy as np, joblib, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load metadata
META = json.loads(Path("meta.json").read_text())
CLASSES = META["class_names"]
CONFIDENCE_FLOOR = 0.45

# --- Logistic Regression ---
VECTORIZER = joblib.load("tfidf_vectorizer.joblib")
LR_MODEL = joblib.load("logistic_regression.joblib")

url_re = re.compile(r"http\S+|www\.\S+")
user_re = re.compile(r"@\w+")
num_re = re.compile(r"\b\d+\b")
punct = str.maketrans("", "", string.punctuation)

def basic_clean(text):
    text = str(text).lower()
    text = url_re.sub(" ", text)
    text = user_re.sub(" ", text)
    text = text.translate(punct)
    text = num_re.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_lr(text):
    X = VECTORIZER.transform([basic_clean(text)])
    probs = LR_MODEL.predict_proba(X)[0]
    return [
        {"label": CLASSES[i], "probability": float(probs[i])}
        for i in range(len(CLASSES))
    ]

# --- BERT ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BERT on {DEVICE}...")
BERT_TOKENIZER = AutoTokenizer.from_pretrained("bert_base")
BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    "bert_base", num_labels=len(CLASSES)
).to(DEVICE)
BERT_MODEL.eval()
print("BERT ready.")

def predict_bert(text):
    enc = BERT_TOKENIZER(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = BERT_MODEL(**enc).logits[0]
    
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return [
        {"label": CLASSES[i], "probability": float(probs[i])}
        for i in range(len(CLASSES))
    ]

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "confidence_floor": CONFIDENCE_FLOOR}

@app.post("/predict/lr")
def lr_endpoint(text: str):
    if not text or len(text.strip()) < 3:
        return {"error": "Text too short"}
    
    dist = predict_lr(text)
    ranked = sorted(dist, key=lambda d: -d["probability"])
    top = ranked[0]
    
    return {
        "distribution": ranked,
        "top": top,
        "model": "Logistic Regression",
        "confidence_low": top["probability"] < CONFIDENCE_FLOOR,
        "confidence_floor": CONFIDENCE_FLOOR
    }

@app.post("/predict/bert")
def bert_endpoint(text: str):
    if not text or len(text.strip()) < 3:
        return {"error": "Text too short"}
    
    dist = predict_bert(text)
    ranked = sorted(dist, key=lambda d: -d["probability"])
    top = ranked[0]
    
    return {
        "distribution": ranked,
        "top": top,
        "model": "BERT Base",
        "confidence_low": top["probability"] < CONFIDENCE_FLOOR,
        "confidence_floor": CONFIDENCE_FLOOR
    }
