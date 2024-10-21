import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import emoji
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Set page background color to black
st.markdown(
    """
    <style>
    .main {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set color palette to PRGn
palette = sns.color_palette("PRGn", 8)

# Load the dataset
file_path = "./data/train.csv"  # Modify as needed for your file path
data = pd.read_csv(file_path)

# Data Cleaning Function
STOP_WORDS = set(stopwords.words('english'))
BANNED_WORDS = ['im', 'one', 'u', 'rt']


def limpiar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.lower()
    texto = emoji.replace_emoji(texto, replace='')
    texto = re.sub(r'[#@&][\S]+', '', texto)
    texto = re.sub(r"http\S+|www\S+|https\S+|[#@']", "", texto, flags=re.MULTILINE)
    texto = texto.encode('ascii', 'ignore').decode('ascii')
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in word_tokenize(texto) if ((word not in STOP_WORDS) and (word not in BANNED_WORDS))]
    return ' '.join(tokens)


data['clean_text'] = data['text'].apply(limpiar_texto)

# Display dataset information
st.header("Dataset Overview")
st.write(data.head())

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

# Tweet target distribution
st.subheader("Tweet Distribution")
fig, ax = plt.subplots()
sns.countplot(x='target', data=data, palette=palette, ax=ax)
ax.set_title('Distribution of Tweet Types')
ax.set_xlabel('Tweet Type')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Additional Visualization: Tweet Length Distribution
st.subheader("Tweet Length Distribution")
data['text_length'] = data['clean_text'].apply(len)
fig, ax = plt.subplots()
sns.histplot(data=data, x='text_length', hue='target', multiple='stack', palette=palette, bins=50, ax=ax)
ax.set_title("Distribution of Tweet Lengths")
ax.set_xlabel("Length of Tweet")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Wordcloud - disaster tweets
st.subheader("Wordclouds")
disaster_tweets = data[data['target'] == 1]['clean_text']
non_disaster_tweets = data[data['target'] == 0]['clean_text']

# Wordcloud - disaster tweets
disaster_words = ' '.join(disaster_tweets)
wordcloud_disaster = WordCloud(width=800, height=400, background_color='black', colormap='Purples').generate(disaster_words)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud_disaster, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud - Disaster Tweets')
st.pyplot(fig)


# Wordcloud - non-disaster tweets
non_disaster_words = ' '.join(non_disaster_tweets)
wordcloud_non_disaster = WordCloud(width=800, height=400, background_color='black', colormap='Greens').generate(non_disaster_words)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud_non_disaster, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud - Non-Disaster Tweets')
st.pyplot(fig)

# Additional Visualization: Top 20 Most Common Words
st.subheader("Top 20 Most Common Words in Disaster and Non-Disaster Tweets")

disaster_tweets_tokens = disaster_tweets.apply(lambda x: word_tokenize(x))
non_disaster_tweets_tokens = non_disaster_tweets.apply(lambda x: word_tokenize(x))

disaster_word_freq = Counter([word for tokens in disaster_tweets_tokens for word in tokens])
non_disaster_word_freq = Counter([word for tokens in non_disaster_tweets_tokens for word in tokens])

# Common words
common_words = set(disaster_word_freq.keys()).intersection(set(non_disaster_word_freq.keys()))
common_words_freq = {word: (disaster_word_freq[word], non_disaster_word_freq[word]) for word in common_words}
common_words_sorted = sorted(common_words_freq.items(), key=lambda x: x[1][0], reverse=True)[:20]

words = [item[0] for item in common_words_sorted]
disaster_freqs = [item[1][0] for item in common_words_sorted]
non_disaster_freqs = [item[1][1] for item in common_words_sorted]

# Bar chart of common words
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(words))
width = 0.35
rects1 = ax.bar(x - width/2, disaster_freqs, width, label='Disaster Tweets', color=palette[3])
rects2 = ax.bar(x + width/2, non_disaster_freqs, width, label='Non-Disaster Tweets', color=palette[6])

ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
ax.set_title('Top 20 Most Common Words in Disaster and Non-Disaster Tweets')
ax.set_xticks(x)
ax.set_xticklabels(words, rotation=90, ha='right')
ax.legend()

st.pyplot(fig)

# Random Forest Model
st.header("Random Forest Model")
st.subheader("Model Results")

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_rf = vectorizer.fit_transform(data['clean_text']).toarray()
y = data['target'].values

# Train-test split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_predictions = rf_model.predict(X_test_rf)

# Model evaluation
st.text("Random Forest Classification Report")
st.text(classification_report(y_test_rf, rf_predictions))

# Confusion matrix
rf_cm = confusion_matrix(y_test_rf, rf_predictions)
fig, ax = plt.subplots()
disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=['Non-Disaster', 'Disaster'])
disp_rf.plot(cmap=plt.cm.Purples, ax=ax)
st.pyplot(fig)

# Sentiment Analysis
st.header("Sentiment Analysis")
analyzer = SentimentIntensityAnalyzer()


def get_vader_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'], sentiment_scores['compound']


data[['neg_sentiment', 'neu_sentiment', 'pos_sentiment', 'compound_sentiment']] = data['clean_text'].apply(
    lambda x: pd.Series(get_vader_sentiment(x))
)

# Sentiment distribution
st.subheader("Sentiment Distribution by Tweet Type")
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=data[data['target'] == 1]['compound_sentiment'], label='Disasters', color=palette[7], fill=True, ax=ax)
sns.kdeplot(data=data[data['target'] == 0]['compound_sentiment'], label='Non-Disasters', color=palette[5], fill=True, ax=ax)
ax.set_title('Compound Sentiment Distribution')
ax.set_xlabel('Compound Sentiment Score')
ax.set_ylabel('Density')
st.pyplot(fig)

# Additional Visualization: Sentiment Score Boxplots
st.subheader("Sentiment Score Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=data, x='target', y='compound_sentiment', palette=palette, ax=ax)
ax.set_title("Boxplot of Compound Sentiment Scores by Tweet Type")
ax.set_xticklabels(['Non-Disaster', 'Disaster'])
ax.set_xlabel('Tweet Type')
ax.set_ylabel('Compound Sentiment Score')
st.pyplot(fig)

# Additional Visualization: Feature Importance from Random Forest
st.subheader("Random Forest Feature Importance")

# Get feature importance from the Random Forest model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features

# Get the corresponding feature names
feature_names = vectorizer.get_feature_names_out()
top_features = [feature_names[i] for i in indices]

# Plot the top 20 feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_features, importances[indices], color=palette[2])
ax.set_title("Top 20 Features by Importance (Random Forest)")
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Features")
st.pyplot(fig)


# Additional Visualization: Correlation Heatmap
st.subheader("Correlation Heatmap")

# Compute correlation matrix
correlation_matrix = data[['neg_sentiment', 'neu_sentiment', 'pos_sentiment', 'compound_sentiment', 'text_length']].corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='PRGn', ax=ax)
ax.set_title("Correlation Heatmap of Sentiments and Tweet Length")
st.pyplot(fig)


# Additional Visualization: ROC Curve
st.subheader("ROC Curve")

# Get the probabilities of the model for the positive class (disaster tweets)
rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_rf, rf_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color=palette[3], lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc="lower right")
st.pyplot(fig)

# Additional Visualization: Distribution of Sentiment by Tweet Length
st.subheader("Sentiment Distribution by Tweet Length")

# Scatter plot of tweet length vs compound sentiment score
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x='text_length', y='compound_sentiment', hue='target', palette=palette, ax=ax)
ax.set_title('Sentiment vs Tweet Length')
ax.set_xlabel('Tweet Length')
ax.set_ylabel('Compound Sentiment Score')
st.pyplot(fig)


# Additional Visualization: Confusion Matrix Heatmap
st.subheader("Confusion Matrix Heatmap")

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='PRGn', cbar=False, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix Heatmap (Random Forest)')
st.pyplot(fig)

# Additional Visualization: Word Length Distribution
st.subheader("Average Word Length Distribution")

# Calculate average word length for each tweet
data['avg_word_length'] = data['clean_text'].apply(lambda x: np.mean([len(word) for word in x.split()]))

# Bar plot for average word length
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=data, x='avg_word_length', hue='target', multiple='stack', palette=palette, bins=50, ax=ax)
ax.set_title('Average Word Length Distribution')
ax.set_xlabel('Average Word Length')
ax.set_ylabel('Frequency')
st.pyplot(fig)
