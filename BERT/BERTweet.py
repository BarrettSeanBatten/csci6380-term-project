import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# Load CSV file containing the tweet data
file_path = '/Users/austinperales/Downloads/csci6380-term-project-data - csci6380-term-project-data.csv'
data = pd.read_csv(file_path, delimiter=',')

# Data Preprocessing: Convert tweets to lowercase and replace non-alphanumeric characters with spaces
data['Tweet'] = data['Tweet'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True)

# Define label encoding dictionary and apply it to convert text labels to integers
label_dict = {'negative': 0, 'positive': 1}
data['Sentiment'] = data['Sentiment'].map(label_dict)

# Check for NaN values in the 'Sentiment' column and handle them by dropping rows with NaN values
if data['Sentiment'].isnull().any():
    print("NaN values found in 'Sentiment'. Handling...")
    data = data.dropna(subset=['Sentiment'])

# Split the data into training and testing sets with 20% of the data reserved for testing
X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data['Sentiment'], test_size=0.2, random_state=42)

# Load the tokenizer and model pre-trained for BERTweet for processing Twitter text
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)

# Tokenize and prepare datasets for training and testing
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

# Custom PyTorch dataset class that prepares batches of tokenized text and labels for model training
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Fetch a single tokenized tweet and its label, and convert them to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        # Return the total number of tweets in the dataset
        return len(self.labels)

# Create instances of the training and testing datasets
train_dataset = TweetDataset(train_encodings, y_train.tolist())
test_dataset = TweetDataset(test_encodings, y_test.tolist())

# Define training parameters
training_args = TrainingArguments(
    output_dir='./results',  # Directory where outputs are saved during training
    num_train_epochs=1,  # Set the number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=32,  # Batch size for evaluation
    max_steps=500,  # Maximum number of training steps to perform
    warmup_steps=50,  # Number of steps to perform learning rate warmup
    weight_decay=0.01,  # Weight decay to apply
    logging_dir='./logs',  # Directory to save logs
    logging_steps=50,  # Log metrics every 50 steps
    evaluation_strategy="steps",  # Evaluate model every logging step
    save_strategy="no",  # Do not save model checkpoints
    load_best_model_at_end=False  # Do not load the best model at the end of training
)

# Initialize the Trainer, configure with model, training args, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Start model training
trainer.train()

# Function to evaluate the model on the test dataset and make predictions
def get_predictions(model, tokenizer, text):
    # Tokenize text, perform model inference, and return the predicted class (0 or 1)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    return probs.argmax()

# Evaluate the model using the test dataset
predictions = [get_predictions(model, tokenizer, tweet) for tweet in X_test]
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Function to predict sentiment of a new tweet
def predict_sentiment(tweet):
    prediction = get_predictions(model, tokenizer, [tweet])
    return 'positive' if prediction == 1 else 'negative'

# Predict sentiment of a sample new tweet and display the result
new_tweet = "The election is gonna be great!"
print("The predicted sentiment of the new tweet is:", predict_sentiment(new_tweet))