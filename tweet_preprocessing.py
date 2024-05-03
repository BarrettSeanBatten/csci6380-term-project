# -*- coding: utf-8 -*-
"""
Created on Fri May  3 01:02:06 2024

@author: Barrett S. Batten 
"""

# Data Preprocessing

import pandas as pd
import re
# pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# Preprocess Text function

def preprocess_text(text):
    """
    Function to preprocess text data:
    - Convert to lowercase
    - Remove URLs
    - Remove special characters
    - Tokenize text
    - Remove stopwords
    """
    # Convert text to lowercase
    # text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove all special characters and digits, but keep alphabetic characters and spaces
    text = re.sub(r'[^a-zA-Z ]+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    # filtered_words = [word for word in tokens]
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]

    
    return " ".join(filtered_words)

# Apply Vader function

def apply_vader(text):
    """
    This function takes a text input, computes its sentiment using
    VADER, and returns the compound score. 
    The compound score is a single measure that describes the 
    overall sentiment of the text, ranging from -1 (most negative)
    to +1 (most positive).
    """
    sid = SentimentIntensityAnalyzer()
    # Return the compound score
    return sid.polarity_scores(text)['compound']  

# Categorize Sentiment function

def categorize_sentiment(score):
    """
    Categorize the sentiment based on the VADER compound score.
    
    Parameters:
    score (float): The compound sentiment score from VADER.
    
    Returns:
    str: The category of sentiment ('positive', 'neutral', 'negative').
    """
    # Original threshold 0.05
    if score > 0.25:
        return 'positive'
    elif score < -0.25:
        return 'negative'
    else:
        return 'neutral'

# Load the raw tweets

df = pd.read_csv('tweets_2024_election_raw.csv', encoding='utf-8')

print("BEFORE DATA PREPROCESSING")
print("Tweet 1: " + df['Tweet'][1])
print("Tweet 2: " + df['Tweet'][2])
print("Tweet 3: " + df['Tweet'][3])
print("Tweet 4: " + df['Tweet'][4])
print("Tweet 5: " + df['Tweet'][5])

# Apply preprocessing to the Tweet column
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Now df['Tweet'] contains the preprocessed text
# Display the first few rows to check the results
print(df['Tweet'].head()) 
print()
print("AFTER DATA PREPROCESSING")
print("Tweet 1: " + df['Tweet'][1])
print("Tweet 2: " + df['Tweet'][2])
print("Tweet 3: " + df['Tweet'][3])
print("Tweet 4: " + df['Tweet'][4])
print("Tweet 5: " + df['Tweet'][5])

# Apply VADER to each tweet and store the results in a new column
df['Sentiment_Score'] = df['Tweet'].apply(apply_vader)

# The 'Sentiment Score' column contains compound scores
# Score ranges from -1 (most extreme negative) to +1 (most extreme positive).

# Display the DataFrame to verify the sentiment scores
print(df.head())

# Apply the categorization function to the sentiment scores
df['Sentiment'] = df['Sentiment_Score'].apply(categorize_sentiment)

# Display the DataFrame to verify the results
print(df[['Tweet', 'Sentiment_Score', 'Sentiment']].head())

# Write DataFrame to CSV file with headers
df.to_csv('csci6380-term-project-data.csv', index=False)

