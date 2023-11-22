import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
import nltk
import re

# Assuming you've already loaded your dataset into 'data'
data = pd.read_csv('bb1.csv')
# data1 = data.drop(['Username', 'Account', 'Post_time'], axis=1)

# Sentiment Analysis
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Replies"] = data["Replies"].apply(lambda x: str(x) if not isinstance(x, str) else x)
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Replies"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Replies"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Replies"]]
data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["Replies"]]
score = data["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05:
        sentiment.append('Positive')
    elif i <= -0.05:
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
data["Sentiment"] = sentiment

# Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(data['Replies'], data['Sentiment'], test_size=0.2, random_state=42)

# Use TfidfVectorizer or CountVectorizer for text data
vectorizer = TfidfVectorizer()  # You can also try CountVectorizer
model = make_pipeline(vectorizer, MultinomialNB())
model.fit(X_train, y_train)
predicted_sentiments = model.predict(X_test)

accuracy = accuracy_score(y_test, predicted_sentiments)
print(f"Accuracy: {accuracy:.2f}")

data['Predicted_Sentiment'] = model.predict(data['Replies'])
print(data[['Replies', 'Sentiment', 'Predicted_Sentiment']])

from sklearn.metrics import confusion_matrix
import seaborn as sns
# Plot confusion matrix
conf_mat = confusion_matrix(y_test, predicted_sentiments, labels=['Positive', 'Neutral', 'Negative'])
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# Predicted sentiments
data['Predicted_Sentiment'] = model.predict(data['Replies'])

# Plotting bar chart
plt.figure(figsize=(14, 6))

# Subplot for bar chart
plt.subplot(1, 2, 1)
sentiment_counts = data['Predicted_Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Predicted Sentiments Bar Chart')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Subplot for pie chart
plt.subplot(1, 2, 2)
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'blue'])
plt.title('Predicted Sentiments Pie Chart')

plt.tight_layout()
plt.show()
