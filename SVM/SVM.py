import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load CSV file (replace with local path)
file_path = '/Users/austinperales/Downloads/csci6380-term-project-data - csci6380-term-project-data.csv'
data = pd.read_csv(file_path, delimiter=',')  # Use comma as the delimiter

# Data Preprocessing
data['Tweet'] = data['Tweet'].str.lower().str.replace('[^\w\s]', ' ')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data['Sentiment'], test_size=0.2, random_state=42)

# Building a Pipeline for the SVM Classifier
model = make_pipeline(CountVectorizer(), TfidfTransformer(), SVC(kernel='linear'))

# Training the model
model.fit(X_train, y_train)

# Predicting the test set results
predicted = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, predicted))
print("Classification Report:\n", classification_report(y_test, predicted))

# Function to predict sentiment of new tweets
def predict_sentiment(tweet):
    return model.predict([tweet])[0]

# Example usage
new_tweet = "The election is gonna be great!"
print("The predicted sentiment of the new tweet is:", predict_sentiment(new_tweet))
