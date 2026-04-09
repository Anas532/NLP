#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# In[4]:


messages = pd.read_csv('SMSSpamCollection.txt', sep='\t',
                           names=["label", "message"])

# sep='\t', t means tab, this was the seperator used in our text file, hence why we used '\t'

# In[6]:


messages

# Message is our input feature, and label is our "Dependent" feature
# 
# 
# 
# We will do some text pre-processing here, we will convert these words into "Vectors"
# 

# In[7]:


messages.shape

# The overall number of text is 5572

# In[12]:


messages["message"].loc[312]

# Hahahahah, crazy

# In[13]:


messages["message"].loc[5570]

# lolll

# In[15]:


#Libraries for Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

# In[17]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# In[22]:


ps = PorterStemmer() ## Used  for stemming

# In[23]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) ## This line is used to remove all the special characters except capital a-z and small a to z
    
    review = review.lower() #lowering so duplicated words not present

    review = review.split() #Splitting, and for each word we traverse it, use stopwords, and then stemming

    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# In[ ]:




# # 1. Creating a Bag of Words model

# In[64]:


from sklearn.feature_extraction.text import CountVectorizer   ## We use countvectorizer to create a Bag of Words
cv = CountVectorizer(max_features=2500, binary = True ,ngram_range= (1,2) )
X = cv.fit_transform(corpus).toarray()

# Binary= If a word is present 2 times in a sentence, or in others words greater than 1, the value will be displayed as one only
# 
# Max_features = 2500 actuallys gives us the top 2500 most occuring features
# 
# ngram_range = (2,2), i'd recommend you to see the documentation for this hyperparameter

# In[65]:


X ## Bag of words created, so this is our input feature

# In[66]:


X.shape

# In[67]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# Just doing label encoding, since if you see above, its just spam and ham

# In[68]:


y

# In[69]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 80/20 split

# In[70]:


X_train

# In[71]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# We use Multinomial Naive Bayes for text classification because it works well with Bag of Words and n-gram features. 
# 
# 
# The text messages are converted into numerical features using CountVectorizer, where each feature represents the frequency of a word or phrase.
# 
# 
# Multinomial Naive Bayes then learns the probability of each word occurring in spam and non-spam messages and uses these probabilities to classify new messages

# In[72]:


#prediction
y_pred=spam_detect_model.predict(X_test)

# In[73]:


from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import classification_report

# In[77]:


score=accuracy_score(y_test,y_pred)
print("score:", score)

# In[75]:


print(classification_report(y_pred,y_test))

# In[ ]:





# # 2. Creating the Tf-Idf model

# In[81]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=2500,ngram_range = (1,2))
X = tv.fit_transform(corpus).toarray()

# TF-IDF improves upon Bag of words by reweighting word frequencies based on their importance across documents. 
# 
# However, both representations remain sparse and do not capture semantic meaning.
# 
# To capture semantic relationships between words, dense word embeddings such as Word2Vec are used

# In[83]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# In[89]:


spam_detect_model = MultinomialNB().fit(X_train, y_train)

# In[90]:


#prediction
y_pred=spam_detect_model.predict(X_test)

# In[91]:


score=accuracy_score(y_test,y_pred)
print("score", score)

# In[94]:


print(classification_report(y_pred,y_test))

# In[ ]:




# #### I'd recommend tho to first apply the train test split, and then apply Tf-idf or BoW, as there can be concerns for data leakage

# ### Using Random forest, MultinomialNB, Logistic Regression  for Tf-idf (OPTIONAL, JUST AN EXPERIMENT)

# In[97]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# In[99]:


rf = RandomForestClassifier(
    n_estimators=300,
    random_state=0,
    n_jobs=-1)

# In[101]:


rf.fit(X_train, y_train)

# In[102]:


y_pred = rf.predict(X_test)

# In[103]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# The Random Forest model uses multiple decision trees to improve classification performance. 
# 
# The number of trees is controlled using n_estimators, while max_depth limits tree complexity to reduce overfitting. 
# 
# The max_features parameter introduces randomness by limiting the number of features considered at each split. 
# 
# Since the dataset is imbalanced, class_weight="balanced" is used to give equal importance to both classes.

# In[ ]:




# In[107]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

models = {
    "MultinomialNB": MultinomialNB(),
    "LogReg": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1)
}

for name, model in models.items():
    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)
    print(name, "acc =", accuracy_score(y_test.ravel(), pred))

# In[ ]:




# # 3. Word2vec

# In[106]:


!pip install gensim

# In[108]:


from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import gensim

# ### Let's import the google pretrained model first, and then we will create our own. This model is trained to create 300 dimensions

# In[109]:


wv = api.load('word2vec-google-news-300') 

# In[114]:


wv['Basketball']

# #### Pretrained model is not bad, impressive actually

# In[ ]:




# ## Applying lemmatization

# In[112]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

# In[115]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# In[116]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess

# #### Simpleprocess converts a document into a list of lowercased tokens, so we dont have to go throught hte hassle of lowering it , manually

# In[118]:


corpus[100]

# In[120]:


words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))

# words ----> Run this in a cell to see the magic for yourself, if i run it, its gonna take a while to scroll down to current cell hhhh

# ### These are the list of our unique words

# In[122]:


model=gensim.models.Word2Vec(words,window=5,min_count=2)

# ### Word2Vec Model Parameters
# 
# 
# model = gensim.models.Word2Vec(words, window=5, min_count=2)
# 
# 
# words: The input corpus, where each sentence is represented as a list of words. The model learns word relationships based on how words co-occur within sentences.
# 
# window = 5: Defines the size of the context window. The model considers up to 5 words before and after a target word, allowing it to capture contextual relationships.
# 
# min_count = 2: Ignores words that appear fewer than 2 times in the corpus. This helps remove rare words and noise, improving training efficiency
# 
# 
# 
# 
# Also by default, the dimension size is 100 since the vector size is 100, Though i recommened you to read the documentation for this as the hyperparameters are explained in much depth

# #### model.wv.index_to_key ---> run this code, to see the unique words, again i am not doing it as i will have to scroll down a lot to reach to the current cell lol

# In[135]:


model.corpus_count ## Our total vocabulary size

# In[136]:


model.epochs

# In[132]:


model.wv.similar_by_word('early')

# In[133]:


#Haha google pretrained model was way better lol

# ## 4. Avg Word2Vec

# In[149]:


import numpy as np

# In[150]:


def avg_word2vec(doc):
    # remove out-of-vocabulary words
    #sent = [word for word in doc if word in model.wv.index_to_key]
    #print(sent)
    
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
                #or [np.zeros(len(model.wv.index_to_key))], axis=0)

# This function converts a document into a single vector by averaging the Word2Vec embeddings of the words it contains.
# 
# doc
# Represents a document as a list of words.
# 
# The function first ignores out-of-vocabulary words, meaning words that are not present in the Word2Vec model’s learned vocabulary.
# 
# For each valid word in the document, the corresponding Word2Vec vector is retrieved from the model.
# 
# All retrieved word vectors are then averaged using np.mean along axis 0, producing a single fixed-length vector that represents the entire document.
# 
# The commented fallback line (using zeros) is intended to handle cases where none of the words in the document exist in the model’s vocabulary, preventing errors.

# In[151]:


!pip install tqdm

# In[152]:


from tqdm import tqdm

# In[ ]:


#apply for the entire sentences
X=[]
for i in tqdm(range(len(words))):
    print("Hello",i)
    X.append(avg_word2vec(words[i]))


# In[154]:


## Run this code and you will see the whole list lol


# In[155]:


type(X)

# In[ ]:



