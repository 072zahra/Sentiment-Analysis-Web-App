import pandas as pd
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv('tweet_emotions.csv')

# Display dataset information
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique sentiments: {df['sentiment'].unique().tolist()}")

# Map the 13 sentiment labels to 3 categories
sentiment_mapping = {
    'happiness': 'happy',
    'love': 'happy',
    'fun': 'happy',
    'relief': 'happy',
    'enthusiasm': 'happy',
    'surprise': 'happy',
    
    'sadness': 'sad',
    'worry': 'sad',
    'boredom': 'sad',
    'empty': 'sad',
    'anger': 'sad',
    'hate': 'sad',
    
    'neutral': 'neutral'
}

# Apply mapping and drop rows with sentiments not in the mapping
df['mapped_sentiment'] = df['sentiment'].map(sentiment_mapping)
df = df.dropna(subset=['mapped_sentiment'])

# Keep only necessary columns
df = df[['content', 'mapped_sentiment']]
df.columns = ['text', 'sentiment']

# Display cleaned dataset information
print(f"Cleaned dataset shape: {df.shape}")
print(f"Unique sentiments after mapping: {df['sentiment'].unique().tolist()}")

# Save cleaned dataset
df.to_csv('dataset/cleaned_data.csv', index=False)
print("Cleaned dataset saved to dataset/cleaned_data.csv")