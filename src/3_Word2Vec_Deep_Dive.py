#!/usr/bin/env python
# coding: utf-8

# In[347]:


# importing the Dataset

import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

# In[81]:


messages

# In[111]:


messages['message'].loc[451]

# In[2]:


#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

# In[3]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# In[97]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# In[ ]:




# In[36]:


corpus

# In[5]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# In[6]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# In[7]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[8]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# In[9]:


#prediction
y_pred=spam_detect_model.predict(X_test)

# In[10]:


from sklearn.metrics import accuracy_score,classification_report

# In[12]:


score=accuracy_score(y_test,y_pred)
print(score)

# In[13]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

# In[14]:


# Creating the TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=2500)
X = tv.fit_transform(corpus).toarray()

# In[15]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[16]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# In[19]:


#prediction
y_pred=spam_detect_model.predict(X_test)

# In[20]:


score=accuracy_score(y_test,y_pred)
print(score)

# In[21]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

# ## Word2vec Implementation

# In[22]:


!pip install gensim

# In[23]:


import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

# In[25]:




vec_king = wv['king']

# In[26]:


vec_king

# In[6]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

# In[7]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# In[8]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess 

# 

# In[9]:


corpus[0]

# In[ ]:


words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))

# In[352]:


words

# In[353]:


import gensim

# In[464]:


### Lets train Word2vec from scratch
model=gensim.models.Word2Vec(words,window=5,min_count=2)

# In[465]:


model.wv.index_to_key

# In[466]:


model.corpus_count

# In[467]:


model.epochs

# In[468]:


model.wv.similar_by_word('kid')

# In[469]:


model.wv['kid'].shape

# In[452]:


def avg_word2vec(doc):
    # remove out-of-vocabulary words
    #sent = [word for word in doc if word in model.wv.index_to_key]
    #print(sent)
    
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
                #or [np.zeros(len(model.wv.index_to_key))], axis=0)
        
    
    

# In[55]:


!pip install tqdm

# In[364]:


from tqdm import tqdm

# In[367]:


words[73]

# In[379]:


type(model.wv.index_to_key)

# In[453]:


#apply for the entire sentences
X=[]
for i in tqdm(range(len(words))):
    print("Hello",i)
    X.append(avg_word2vec(words[i]))

    

# In[454]:


type(X)

# In[455]:


X_new=np.array(X)

# In[456]:


X_new[3]

# In[ ]:




# In[ ]:




# In[459]:


X_new.shape

# In[ ]:



