#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
import warnings
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


# Reading in the amazon reviews as a pandas frame and only including review and rating columns

file_name = "amazon_reviews_us_Office_Products_v1_00.tsv"
df = pd.read_csv(file_name, sep='\t', usecols=['review_body', 'star_rating'], low_memory=False)

# Dropping all NAs in the dataframe

df.dropna(inplace=True)

# Rename columns Review and Rating

df.rename(columns={'review_body': 'Review', 'star_rating': 'Rating'}, inplace=True)

# Transform the Rating column to a number

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')


# Create a new column in the dataframe called ‘Sentiment’ 
# Ratings over 3 are identified as positive reviews with value 1
# Ratings below 3 are identified as negative reviews with value 0
# Ratings at 3 are identified as neutral reviews with value 3

df['Sentiment'] = np.where(df['Rating'] > 3, 1, np.where(df['Rating'] < 3, 0, 3))

#Count each type of sentiment and print

sentiment_counts = df['Sentiment'].value_counts()
print("Positive reviews:", sentiment_counts.get(1, 0), ", Negative reviews:", sentiment_counts.get(0, 0),
      ", Neutral reviews:", sentiment_counts.get(3, 0))

#Remove all neutral reviews

df = df[(df['Rating'] != 3)]

# Filter the dataframe to 100,000 positive reviews and 100,000 negative reviews

pos_reviews = df[df['Sentiment'] == 1].sample(n=100000, random_state=42)
neg_reviews = df[df['Sentiment'] == 0].sample(n=100000, random_state=42)

filtered_df = pd.concat([pos_reviews, neg_reviews])

# Convert to lowercase
# Remove HTML and URLS
# Remove non-alphabetical characters
# Remove extra spaces
filtered_df['Original Review'] = filtered_df['Review']
filtered_df['Review'] = filtered_df['Review'].str.lower()
filtered_df['Review'] = filtered_df['Review'].fillna('').astype(str)
def clean(review):
    if "<" in review and ">" in review:
        review = BeautifulSoup(review, "html.parser").get_text()
    review = re.sub(r'http\S+|www\S+|https\S+', '', review)
    review = re.sub(r'<[^>]*>', '', review)
    review = re.sub(r'[^a-zA-Z\s]', '', review) 
    return review
filtered_df['Review'] = filtered_df['Review'].apply(clean)
filtered_df['Review'] = filtered_df['Review'].str.strip() 
filtered_df['Review'] = filtered_df['Review'].str.replace(r'\s+', ' ', regex=True)


# In[7]:


# Create a dictionary of contractions

contractions_dict = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "might've": "might have",
    "mustn't": "must not",
    "must've": "must have",
    "needn't": "need not",
    "o'clock": "of the clock",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}


# Apply contraction dictionary to the reviews column

contract = re.compile('|'.join(re.escape(key) for key in contractions_dict.keys()))

def expand(text):
    return contract.sub(lambda x: contractions_dict[x.group()], text)

filtered_df['Review'] = filtered_df['Review'].apply(expand)

# Print out the average review length before and after cleaning

before_avg = filtered_df['Original Review'].apply(len).mean()
after_avg = filtered_df['Review'].apply(len).mean()

print("Average review length before cleaning:", before_avg, ", Average review length after cleaning: ", after_avg)


# Import necessary NLTK packages

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# Save the column at the moment for future statistics

filtered_df['Before Changes'] = filtered_df['Review']

# Remove stop words

def stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    return ' '.join([word for word in words if word.lower() not in stop_words])
filtered_df['Review'] = filtered_df['Review'].apply(stop_words)


# Import necessary libraries and packages

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Map tags to parts of speech to help with lemmatization

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'): 
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatize function that includes parts of speech

def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    
    return ' '.join(lemmatized)

# Apply lematization to the Reviews column

filtered_df['Review'] = filtered_df['Review'].apply(lemmatization)

# Print the average length of reviews before and after preprocessing
before = filtered_df['Before Changes'].apply(len).mean()
after = filtered_df['Review'].apply(len).mean()
print("Avg length before preprocessing:", before, ", Avg length after preprocessing:", after)


# Import necessary packages

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Create a TFIDF Function

lets_vectorize = TfidfVectorizer(
    # Removes stop words
    stop_words ='english',

    # Function will consider groups of 1 word and 2 words as features
    ngram_range=(1,2)
)

# Builds a vocabulary of unique terms and assigns a TFIDF score to each
features = lets_vectorize.fit_transform(filtered_df['Review'])

# Labels (or y value) will be the Sentiment score
labels = filtered_df['Sentiment']

# Divides the data into training and testing sets where training is 80% of the data and testing is 20%
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state=16)


# Import necessary libraries from sklearn

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize a perceptron model and then fit to your training data
mr_perceptron = Perceptron(max_iter=100000, tol=1e-3, random_state=16)
mr_perceptron.fit(features_train, labels_train)

# Make predictions based on the training and testing datasets
labels_train_pred = mr_perceptron.predict(features_train)
labels_test_pred = mr_perceptron.predict(features_test)

# Calculate scores for all the training predictions
train_accuracy = accuracy_score(labels_train, labels_train_pred)
train_precision = precision_score(labels_train, labels_train_pred)
train_recall = recall_score(labels_train, labels_train_pred)
train_f1 = f1_score(labels_train, labels_train_pred)

# Calculate scores for all the testing predictions
test_accuracy = accuracy_score(labels_test, labels_test_pred)
test_precision = precision_score(labels_test, labels_test_pred)
test_recall = recall_score(labels_test, labels_test_pred)
test_f1 = f1_score(labels_test, labels_test_pred)

# Print scores
print("Perceptron- Train_Accuracy:", train_accuracy, "Train_Precision:", train_precision, "Train_Recall:",
      train_recall, "Train_F1:", train_f1, "Test_Accuracy:", test_accuracy, "Test_Precision:", test_precision,
      "Test_Recall:", test_recall, "Test_F1: ", test_f1)


# Import necessary libraries from sklearn
from sklearn.svm import LinearSVC

# Initialize an SVM model and then fit to your training data
svm_model = LinearSVC(random_state=16, max_iter=100000)
svm_model.fit(features_train, labels_train)

# Make predictions based on the training and testing datasets
labels_train_pred = svm_model.predict(features_train)
labels_test_pred = svm_model.predict(features_test)

# Calculate scores for all the training predictions
train_accuracy = accuracy_score(labels_train, labels_train_pred)
train_precision = precision_score(labels_train, labels_train_pred)
train_recall = recall_score(labels_train, labels_train_pred)
train_f1 = f1_score(labels_train, labels_train_pred)

# Calculate scores for all the testing predictions
test_accuracy = accuracy_score(labels_test, labels_test_pred)
test_precision = precision_score(labels_test, labels_test_pred)
test_recall = recall_score(labels_test, labels_test_pred)
test_f1 = f1_score(labels_test, labels_test_pred)

# Print  scores
print("SVM- Train_Accuracy:", train_accuracy, "Train_Precision:", train_precision, "Train_Recall:",
      train_recall, "Train_F1:", train_f1, "Test_Accuracy:", test_accuracy, "Test_Precision:", test_precision,
      "Test_Recall:", test_recall, "Test_F1:", test_f1)


# Import necessary libraries from sklearn
from sklearn.linear_model import LogisticRegression

# Initialize an Logistic Regression model and then fit to your training data
log_reg = LogisticRegression(random_state=16, max_iter=5000)
log_reg.fit(features_train, labels_train)

# Make predictions based on the training and testing datasets
labels_train_pred = log_reg.predict(features_train)
labels_test_pred = log_reg.predict(features_test)

# Calculate scores for all the training predictions
train_accuracy = accuracy_score(labels_train, labels_train_pred)
train_precision = precision_score(labels_train, labels_train_pred)
train_recall = recall_score(labels_train, labels_train_pred)
train_f1 = f1_score(labels_train, labels_train_pred)

# Calculate scores for all the testing predictions
test_accuracy = accuracy_score(labels_test, labels_test_pred)
test_precision = precision_score(labels_test, labels_test_pred)
test_recall = recall_score(labels_test, labels_test_pred)
test_f1 = f1_score(labels_test, labels_test_pred)

# Print  scores
print("Logistic Regression- Train_Accuracy:", train_accuracy, "Train_Precision:", train_precision, "Train_Recall:",
      train_recall, "Train_F1:", train_f1, "Test_Accuracy:", test_accuracy, "Test_Precision:", test_precision,
      "Test_Recall:", test_recall, "Test_F1:", test_f1)


# Import necessary libraries from sklearn
from sklearn.naive_bayes import MultinomialNB

# Initialize a Naive Bayes model and then fit to your training data
nb_model = MultinomialNB()
nb_model.fit(features_train, labels_train)

# Make predictions based on the training and testing datasets
labels_train_pred = nb_model.predict(features_train)
labels_test_pred = nb_model.predict(features_test)

# Calculate scores for all the training predictions
train_accuracy = accuracy_score(labels_train, labels_train_pred)
train_precision = precision_score(labels_train, labels_train_pred)
train_recall = recall_score(labels_train, labels_train_pred)
train_f1 = f1_score(labels_train, labels_train_pred)

# Calculate scores for all the testing predictions
test_accuracy = accuracy_score(labels_test, labels_test_pred)
test_precision = precision_score(labels_test, labels_test_pred)
test_recall = recall_score(labels_test, labels_test_pred)
test_f1 = f1_score(labels_test, labels_test_pred)

# Print training scores
print("Naive Bayes- Train_Accuracy:", train_accuracy, "Train_Precision:", train_precision, "Train_Recall:",
      train_recall, "Train_F1:", train_f1, "Test_Accuracy:", test_accuracy, "Test_Precision:", test_precision,
      "Test_Recall:", test_recall, "Test_F1:", test_f1)

