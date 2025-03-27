#!/usr/bin/env python
# coding: utf-8

# ### SMS SPAM COLLECTION DATASET

# In[1]:


#required libraies 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


#loading the dataset 
df=pd.read_csv(r"C:\Users\sirim\Downloads\spam.csv",encoding='latin-1')


# #### basic information about data

# In[4]:


df.head()


# In[5]:


df.isna().sum()


# In[6]:


df.duplicated().sum()


# #### DATA PREPROCESSING:

# In[7]:


# removing unnecessary columns
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Convert labels to binary values (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Display first few rows
print(df.head())


# In[8]:


df.head()


# #### TEXT CLEANING:

# In[9]:


# Define def function to clean the text at once 
def clean_all_text(text):
    text = text.lower()  # LOWERCASE
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Applying  text cleaning function to the message column
df['clean_message'] = df['message'].apply(clean_text)

# print first few cleaned messages
print(df[['message', 'clean_message']].head())


# In[10]:


df.head()


# #### spliting the dataset into training and testing, (80%training, 20% testing)

# In[11]:


# spliting the train and testing
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)


# ####  Convert Text to Numerical Data Using TF-IDF

# In[12]:


# Converting  text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# #### Train a Machine Learning Model

# In[13]:


# use Train Naïve Bayes(multinomial) classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# #### predictions

# In[14]:


# Predict the messages spam/ham 
y_pred = model.predict(X_test_tfidf)


# #### Evaluate the Model

# In[24]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Display confusion matrix as a heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


#COUNT PLOT
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(x=df['label'])
plt.title("Spam vs. Ham Count")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.xticks(ticks=[0,1], labels=["Ham", "Spam"])
plt.show()


# In[36]:


#text length distribution
plt.figure(figsize=(8,8))
sns.histplot(df[df['label'] == 1]['message'].apply(len), bins=30, kde=True, color='blue', label="Spam")
sns.histplot(df[df['label'] == 0]['message'].apply(len), bins=30, kde=True, color='pink', label="Ham")
plt.legend()
plt.title("Distribution of Message Lengths")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Frequency")
plt.show()


# In[ ]:





# In[26]:


get_ipython().system('pip install WordCloud')


# ####  Visualizing Frequent Words Using WordCloud

# In[27]:


from wordcloud import WordCloud

# Spam messages word cloud
spam_words = " ".join(df[df['label'] == 1]['clean_message'])
spam_wordcloud = WordCloud(width=600, height=400, background_color='black').generate(spam_words)

plt.figure(figsize=(8, 6))
plt.imshow(spam_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Spam Messages")
plt.show()

# Ham messages word cloud
ham_words = " ".join(df[df['label'] == 0]['clean_message'])
ham_wordcloud = WordCloud(width=600, height=400, background_color='black').generate(ham_words)

plt.figure(figsize=(8, 6))
plt.imshow(ham_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Ham Messages")
plt.show()


# In[ ]:




