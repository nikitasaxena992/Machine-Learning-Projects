#!/usr/bin/env python
# coding: utf-8

# ### SENTIMENT ANALYSIS
# 
# Sentiment analysis is widely applied to voive of the customer material such as reviews and survey responses,online and social media and healthcare materials for application that range from marketing to customer service to clinical medicine.  In this project we are analyzing the zomato review based on the food.

# In[5]:


### importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px   


# In[6]:


### reading csv
df = pd.read_csv("E:/MyLearnings/Machine_learning/SentimentAnalysis/data/Reviews.csv")
df.head()


# In[7]:


### We are doing visualization here by plottoing bar graph an the basis of score which is give in dataset..
fig = px.histogram(df, x= "Score")
fig.update_traces(marker_color = "cyan", marker_line_color = 'rgb(8,48,107)',marker_line_width = 1.5)
fig.update_layout(title_text = "Product Score")
fig.show()


# In[8]:


from nltk.corpus import stopwords
from wordcloud import WordCloud 
from nltk.corpus import words
from nltk.corpus import *


# In[9]:


from nltk.corpus import names
from nltk.classify import apply_features


# In[10]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[11]:


stop = set(stopwords.words('english'))


# In[ ]:


# Here we are craeting word cloud of the whole revi ew text
# stop = set(stopwords.words('english'))
stopwords.update(["br","href"]) 
text = " ".join(review for review in df.Text)
wordcloud = WordCloud(stopwords = stopwords).generate(text)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")             
plt.savefig("word.png")
plt.show()                  


# In[12]:


### Here we have made chnges if the score is greater than 3 then we have converted into 1 and for less than 3 we  have converted it into -1
df = df[df["Score"] !=3]
df['sentiment'] = df['Score'].apply(lambda rating: +1 if rating > 3 else -1)


# In[13]:


df['sentiment']


# In[14]:


### We have intialized 1 for positive sentiment and -1 for negative sentiments
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]


# In[15]:


positive


# In[16]:


negative


# In[17]:


stop = set(stopwords.words('english'))


# In[18]:


nltk.download('stopwords')


# In[19]:


### Here we have counting the total number of stopwords
stop


# In[20]:


import nltk


# In[21]:


stop = set(stopwords.words('english'))


# In[ ]:


### Creating word cloud for positive text reviews
stopwords.update(["br","href","good","great"])
pos = " ".join(review for review in positive.Summary)
wordcloud2 = WordCloud(stopwords = stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation = 'bilinear')
plt.axis("off")
plt.show()      


# In[22]:


### Creating positive and negative histogram graphs
df['sentiment'] = df['sentiment'].replace({-1: 'negative'})
df['sentiment'] = df['sentiment'].replace({1:'positive'})
fig = px.histogram(df, x = "sentiment")
fig.update_layout(title_text = "Product Sentiment")
fig.show()


# In[25]:


### Removing punctuation from text
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?",".",";",":","!",'"'))
    return final
df['Text'] = df['Text'].apply(remove_punctuation)
df = df.dropna(subset=['Summary'])
df['Summary'] = df['Summary'].apply(remove_punctuation)
######## After removing punctuations we are having text
df['Summary']


# In[26]:


## Initalizing the summary and sentiments in new dataframe thats is dfNew
dfNew = df[['Summary','sentiment']]
dfNew.head() 


# In[27]:


## Now we will have machine learning model for that we have dived the random test and train data
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <=0.8]
test = df[df['random_number'] > 0.8]


# In[28]:


### we have train the summary wiyh countvectorizer and same doing with test summary
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])


# In[31]:


### Implementing logistic regression                                  
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[32]:


X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']

y_test = test['sentiment']


# In[33]:


lr.fit(X_train , y_train)


# In[34]:


### calculating the predictions 
predictions = lr.predict(X_test)


# In[35]:


predictions


# In[36]:


from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)


# In[37]:


print(classification_report(predictions,y_test))


# In[ ]:




