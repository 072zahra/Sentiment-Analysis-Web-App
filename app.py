from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import sqlite3
import os

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS sentiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Load model
def load_model():
    model_path = 'model/saved_model'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    classes = np.load(f'{model_path}/classes.npy', allow_pickle=True)
    return tokenizer, model, classes

# Predict sentiment
def predict_sentiment(text, tokenizer, model, classes):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            token_type_ids=encoding['token_type_ids']
        )
        
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        
    return classes[preds.item()]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    text = request.form.get('text')
    
    if not name or not text:
        return jsonify({'error': 'Name and text are required'}), 400
    
    try:
        tokenizer, model, classes = load_model()
        sentiment = predict_sentiment(text, tokenizer, model, classes)
        
        # Save to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('INSERT INTO sentiments (name, text, sentiment) VALUES (?, ?, ?)',
                  (name, text, sentiment))
        conn.commit()
        conn.close()
        
        return render_template('result.html', name=name, text=text, sentiment=sentiment)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    app.run(debug=True) 