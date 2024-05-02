# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:06:29 2024

@author: Barrett Sean Batten

Goal: Experiment with various methods: data preprocessing,
data labeling, and keyword extraction. Findings from this file
will be incorporated into main files in the parent directory.
This is largely a test file.

At time of writing, the raw data consists of 500 tweets by 
querying the Twitter API via candidate keywords. Here,
we use keyword extraction to see if there exists better keywords
and we also consider hashtags for the final data collection
process.

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

# Custom Join function

def custom_join(tokens):
    """
    Join tokens into a string, ensuring that '@' and '#' are not detached from their following words.
    """
    output = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in {'@', '#'} and i + 1 < len(tokens):
            output.append(token + tokens[i + 1])  # Combine '@' or '#' with the next token
            skip_next = True  # Skip the next token since it's already included
        else:
            output.append(token)
    return ' '.join(output)

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
    
    # Remove special characters and digits, but keep "@" and "#"
    text = re.sub(r'[^a-zA-Z@# ]', '', text)
    # Remove space after @ or #
    text = re.sub(r'([@#])\s+', r'\1', text)  
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    # filtered_words = [word for word in tokens]
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]

    
    return custom_join(filtered_words)

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
    
# Extract Keywords function

def extract_keywords(corpus, top_n=10):
    """
    Extract the top_n keywords from a corpus using TF-IDF.
    
    Parameters:
    corpus (list): A list of strings, each representing a document.
    top_n (int): The number of top keywords to return.
    
    Returns:
    list: Top N keywords based on their TF-IDF scores.
    """
    # Initialize TF-IDF Vectorizer with English stopwords
    tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
    
    # Fit and transform the corpus
    tfidf_matrix = tfidf.fit_transform(corpus)
    
    # Get feature names to use as DataFrame column headers
    feature_names = tfidf.get_feature_names_out()
    
    # Get the dense version of matrix to easily find max tf-idf values
    dense = tfidf_matrix.todense()
    
    # Get the mean of the tfidf values for each term across all documents
    term_mean_scores = dense.mean(axis=0).tolist()[0]
    
    # Pair feature names with their corresponding scores
    term_scores = zip(feature_names, term_mean_scores)
    
    # Sort the terms by their score in descending order
    sorted_term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)
    
    # Return the top_n terms with the highest mean scores
    return [term for term, score in sorted_term_scores[:top_n]]

# Load the data into a DataFrame
df = pd.read_csv("data_subset.csv")

# print("BEFORE DATA PREPROCESSING")
# print("Tweet 1: " + df['Tweet'][1])
# print("Tweet 2: " + df['Tweet'][2])
# print("Tweet 3: " + df['Tweet'][3])
# print("Tweet 4: " + df['Tweet'][4])
# print("Tweet 5: " + df['Tweet'][5])
# I want to see how preprocessing handles # character
# print("Tweet 31: " + df['Tweet'][31])
# print("Tweet 33: " + df['Tweet'][33])
# print("Tweet 34: " + df['Tweet'][34])

# Apply preprocessing to the Tweet column
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Now df['Tweet'] contains the preprocessed text
# Display the first few rows to check the results
# print(df['Tweet'].head()) 
# print()
# print("AFTER DATA PREPROCESSING")
# print("Tweet 1: " + df['Tweet'][1])
# print("Tweet 2: " + df['Tweet'][2])
# print("Tweet 3: " + df['Tweet'][3])
# print("Tweet 4: " + df['Tweet'][4])
# print("Tweet 5: " + df['Tweet'][5])
# I want to see how preprocessing handles # character
# print("Tweet 31: " + df['Tweet'][31])
# print("Tweet 33: " + df['Tweet'][33])
# print("Tweet 34: " + df['Tweet'][34])

# Apply VADER to each tweet and store the results in a new column
df['Sentiment_Score'] = df['Tweet'].apply(apply_vader)

# The 'Sentiment Score' column contains compound scores
# Score ranges from -1 (most extreme negative) to +1 (most extreme positive).

# Display the DataFrame to verify the sentiment scores
# print(df.head())

# Apply the categorization function to the sentiment scores
df['Sentiment'] = df['Sentiment_Score'].apply(categorize_sentiment)

# Display the DataFrame to verify the results
print(df[['Tweet', 'Sentiment_Score', 'Sentiment']].head())

# Write DataFrame to CSV file with headers
df.to_csv('preproc_data_subset.csv', index=False)

# Corpus

# Create a corpus for positive sentiment tweets
positive_corpus = df[df['Sentiment'] == 'positive']['Tweet'].tolist()

# Create a corpus for negative sentiment tweets
negative_corpus = df[df['Sentiment'] == 'negative']['Tweet'].tolist()

# Create a corpus for neutral sentiment tweets
neutral_corpus = df[df['Sentiment'] == 'neutral']['Tweet'].tolist()

print("Number of tweets in the positive corpus:", len(positive_corpus))
print("Number of tweets in the negative corpus:", len(negative_corpus))
print("Number of tweets in the neutral corpus:", len(neutral_corpus))

# Keyword Extraction

keywords_positive = extract_keywords(positive_corpus, top_n=10)
keywords_negative = extract_keywords(negative_corpus, top_n=10)
keywords_neutral = extract_keywords(neutral_corpus, top_n=10)

print("Top Positive Keywords:", keywords_positive)
print("Top Negative Keywords:", keywords_negative)
print("Top Neutral Keywords:", keywords_neutral)