import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# Load CSV file
file_path = '/Users/austinperales/Downloads/csci6380-term-project-data - csci6380-term-project-data.csv'
data = pd.read_csv(file_path, delimiter=',')

# Data Preprocessing
data['Tweet'] = data['Tweet'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True)

# Define label encoding and apply it
label_dict = {'negative': 0, 'positive': 1}
data['Sentiment'] = data['Sentiment'].map(label_dict)

# Check for NaN values in 'Sentiment' and handle them
if data['Sentiment'].isnull().any():
    print("NaN values found in 'Sentiment'. Handling...")
    data = data.dropna(subset=['Sentiment'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data['Sentiment'], test_size=0.2, random_state=42)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenization and dataset preparation
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

# Handles batches of data for training and testing
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, y_train.tolist())
test_dataset = TweetDataset(test_encodings, y_test.tolist())

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Keeping it to one epoch to control the duration
    per_device_train_batch_size=16,  # Increased batch size for speed, adjust based on your hardware
    per_device_eval_batch_size=32,  # Larger batch size for quicker evaluation
    max_steps=500,  # Limiting the steps further if needed, adjust based on previous run durations
    warmup_steps=50,  # Adjusted warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,  # Log every 50 steps to monitor the progress
    evaluation_strategy="steps",  # Evaluate periodically, can set to 'epoch' if it runs too frequently
    save_strategy="no",  # To save time by not saving any models, change if model checkpointing is desired
    load_best_model_at_end=False  # To save time by not loading the best model at the end
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluation and prediction
def get_predictions(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    return probs.argmax()

predictions = [get_predictions(model, tokenizer, tweet) for tweet in X_test]
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Predict sentiment of new tweets
def predict_sentiment(tweet):
    prediction = get_predictions(model, tokenizer, [tweet])
    return 'positive' if prediction == 1 else 'negative'

new_tweet = "The election is gonna be great!"
print("The predicted sentiment of the new tweet is:", predict_sentiment(new_tweet))