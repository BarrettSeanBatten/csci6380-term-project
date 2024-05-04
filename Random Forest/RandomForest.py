import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load CSV file (replace with local path)
file_path = '/Users/austinperales/Downloads/csci6380-term-project-data - csci6380-term-project-data.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data['Tweet'] = data['Tweet'].str.lower().str.replace('[^\w\s]', ' ')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data['Sentiment'], test_size=0.2, random_state=42)

# Define a pipeline combining a text feature extractor with a Random Forest classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(random_state=42)),
])

# Define parameter grid to search
parameter_grid = {
    'clf__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'clf__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'clf__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'tfidf__use_idf': (True, False),  # Optionally adding a parameter for the TfidfTransformer
    'vect__ngram_range': [(1, 1), (1, 2)],  # Using unigrams or bigrams
}

# Create a grid search object with the defined pipeline and parameters
grid_search = GridSearchCV(pipeline, parameter_grid, cv=5, verbose=1, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predicting the test set results using the best model
predicted = best_model.predict(X_test)

# Evaluating the best model
print("Best Model Parameters:", grid_search.best_params_)
print("Best Model Accuracy:", accuracy_score(y_test, predicted))
print("Classification Report:\n", classification_report(y_test, predicted))

# Function to predict sentiment of new tweets using the best model
def predict_sentiment(tweet):
    return best_model.predict([tweet])[0]

# Example usage
new_tweet = "The election is gonna be great!"
print("The predicted sentiment of the new tweet is:", predict_sentiment(new_tweet))
