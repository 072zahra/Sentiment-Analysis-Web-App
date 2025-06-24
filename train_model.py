import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += len(labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader), correct_predictions.double() / total_predictions

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader), correct_predictions.double() / total_predictions

def main():
    # Parameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('dataset/cleaned_data.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Encode labels
    le = LabelEncoder()
    df['encoded_sentiment'] = le.fit_transform(df['sentiment'])
    print(f"Sentiment mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Save label encoder classes for inference
    np.save('model/saved_model/classes.npy', le.classes_)
    
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = SentimentDataset(
        texts=train_df['text'].values,
        labels=train_df['encoded_sentiment'].values,
        tokenizer=tokenizer
    )
    
    test_dataset = SentimentDataset(
        texts=test_df['text'].values,
        labels=test_df['encoded_sentiment'].values,
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(le.classes_)
    )
    model.to(device)
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device
        )
        
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        
        val_loss, val_acc = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device
        )
        
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save model
            os.makedirs('model/saved_model', exist_ok=True)
            model.save_pretrained('model/saved_model')
            tokenizer.save_pretrained('model/saved_model')
            print("Model saved!")
            
    print(f"Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main() 