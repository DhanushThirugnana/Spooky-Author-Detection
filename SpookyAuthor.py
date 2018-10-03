
# coding: utf-8

# In[18]:

import numpy as np
import csv
import pandas as pd
import operator as op
import nltk
nltk.download
from nltk.stem import WordNetLemmatizer
import re
import string
import sklearn.feature_extraction.text
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


# In[2]:

train1 = pd.read_csv('train.csv',encoding = "ISO-8859-1")
valid = pd.read_csv('validation.csv',encoding = "ISO-8859-1")
test = pd.read_csv('test.csv',encoding = "ISO-8859-1")
text1 = train1.loc[:,"text"]
text_target = train1.loc[:,"author"]
valid_data1 = valid.loc[:,"text"]
valid_target = valid.loc[:,"author"]
test_data1 = test.loc[:,"text"]


# In[3]:

e_ct = 0
h_ct = 0
m_ct = 0
author_count = train1.loc[:,"author"]
for i in range(0,len(train1)):
    if(author_count[i] == "EAP"):
        e_ct = e_ct+1
    if(author_count[i] == "HPL"):
        h_ct = h_ct+1
    if(author_count[i] == "MWS"):
        m_ct = m_ct+1
count = [e_ct,h_ct,m_ct]
author = ['EAP','HPL','MWS']
y_pos = np.arange(len(author))
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, author)
plt.ylabel('number of sentences in training dataset')
plt.show()


# In[4]:

a = list()
text = train1.loc[:,"text"]
for j in range(0,len(train1)):
    a += re.findall(r"[\w']+|[.,!?;]", text[j])
allwords = a 


# In[5]:

frequency = {}
for word in allwords:
    count = frequency.get(word,0)
    frequency[word] = count + 1


# In[6]:

sorted_d = sorted(frequency.items(), key=op.itemgetter(1),reverse=True)
freq = sorted_d


# In[7]:

x = list()
for z in range(0,50): 
    x.append(sorted_d[z][1])
y = list()
for z in range(0,50): 
    y.append(sorted_d[z][0])


# In[8]:


plt.figure(figsize=(30, 20))
y_pos = np.arange(len(y))
plt.barh(y_pos, x)
plt.yticks(y_pos, y)
plt.ylabel('frequency of words')
plt.xlabel('top 50 words as per frequency')
plt.show()   


# In[9]:

bt = len(sorted_d)
bitch = list()
for z in range(bt-50,bt): 
    bitch.append(sorted_d[z][0])
x = list()
z = 0
for z in range(bt-50,bt): 
    x.append(sorted_d[z][1])
plt.figure(figsize=(40, 30))
y_pos = np.arange(len(bitch))
plt.barh(y_pos, x)
plt.yticks(y_pos, bitch)
plt.ylabel('frequency of words')
plt.xlabel('bottom 50 words as per frequency')
plt.show()   


# In[10]:

def lemmat(l,y):
    b = []
    for x in l:
        if x not in y:
            b.append(x)
    lmtzr = WordNetLemmatizer()
    lemmatized_words = [lmtzr.lemmatize(x) for x in b]
    return lemmatized_words


# In[11]:

lemm_a = lemmat(allwords,y) 


# In[12]:

text = []
sentence = []
a = []
for j in text1:
    a = re.findall(r"[\w']+|[.,!?;]", j)
    lemm_t = lemmat(a,y)
    sentence = ' '. join(lemm_t)
    text.append(sentence)


# In[13]:

valid_data = []
sentence1 = []
a = []
for j in valid_data1:
    a = re.findall(r"[\w']+|[.,!?;]", j)
    lemm_t = lemmat(a,y)
    sentence1 = ' '. join(lemm_t)
    valid_data.append(sentence1)


# In[14]:

test_data = []
sentence1 = []
a = []
for j in test_data1:
    a = re.findall(r"[\w']+|[.,!?;]", j)
    lemm_t = lemmat(a,y)
    sentence1 = ' '. join(lemm_t)
    test_data.append(sentence1)


# In[15]:

e = []
h=[]
m=[]
author_count = train1.loc[:,"author"]
text = train1.loc[:,"text"]
for i in range(0,len(train1)):
    if(author_count[i] == "EAP"):
        e += re.findall(r"[\w']+|[.,!?;]", text[i])
    if(author_count[i] == "HPL"):
        h += re.findall(r"[\w']+|[.,!?;]", text[i])
    if(author_count[i] == "MWS"):
        m += re.findall(r"[\w']+|[.,!?;]", text[i])


# In[16]:

#freqe = frequencyfunc(e)
#freqh = frequencyfunc(h)
#freqm = frequencyfunc(m)
lemm_e = lemmat(e,y)
lemm_h = lemmat(h,y)
lemm_m = lemmat(m,y)


# In[ ]:

def result(file):
    probability_e = []
    probability_h = []
    probability_m = []
    result_nb = []
    z = 0
    a = []
    for j in file:
        prob_e = 1
        prob_h = 1
        prob_m = 1
        b = 0
        a = re.findall(r"[\w']+|[.,!?;]", j)
        for i in a:
            if lemm_a.count(i) != 0:
                prob_e = prob_e * ((lemm_e.count(i)/lemm_a.count(i))+0.0001)
                #prob_e = prob_e * ((e.count(i)/allwords.count(i))+0.0001)
                prob_h = prob_h * ((lemm_h.count(i)/lemm_a.count(i))+0.0001)
                #prob_h = prob_h * ((h.count(i)/allwords.count(i))+0.0001)
                prob_m = prob_m * ((lemm_m.count(i)/lemm_a.count(i))+0.0001)
                #prob_m = prob_m * ((m.count(i)/allwords.count(i))+0.0001)
        if max(prob_e,prob_h,prob_m) == prob_e:
            b = "EAP"
        if max(prob_e,prob_h,prob_m) == prob_h:
            b = "HPL"
        if max(prob_e,prob_h,prob_m) == prob_m:
            b = "MWS"
        probability_e.append(prob_e)
        probability_h.append(prob_h)
        probability_m.append(prob_m)
        result_nb.append(b)
        z = z+1
    return result_nb, probability_e, probability_h, probability_m
    


# In[ ]:

result_nb,probability_e,probability_h,probability_m = result(valid_data)


# In[ ]:

result_nbc,probability_eap,probability_hpl,probability_mws = result(test_data)


# In[ ]:

call = 0
ctrue = 0
l = np.array(valid_target)
m = np.array(result_nb)
for i in range(len(l)):
    call += 1
    if l[i] == m[i]:
        ctrue += 1
vacc_nb = ctrue / call


# In[28]:

from sklearn.naive_bayes import MultinomialNB
classifier = Pipeline([('v', CountVectorizer()), ('t', TfidfTransformer()), ('c', MultinomialNB()),])
classifier.fit(text, text_target)
predicted_nb = classifier.predict(valid_data)  


# In[29]:

met_nb = (metrics.classification_report(valid_target, predicted_nb ,digits = 2))
cmat_nb = metrics.confusion_matrix(valid_target, predicted_nb)


# In[30]:

from sklearn.linear_model import SGDClassifier
classifier = Pipeline([('v', CountVectorizer()),('t', TfidfTransformer()),('c', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, random_state=30))])
classifier.fit(text, text_target)  
predicted_svm = classifier.predict(valid_data)
vacc_svm = np.mean(predicted_svm == valid_target)          
result_svm = classifier.predict(test_data)


# In[31]:

met_svm = (metrics.classification_report(valid_target, predicted_svm ,digits = 2))
cmat_svm = metrics.confusion_matrix(valid_target, predicted_svm)


# In[32]:

from sklearn.neighbors import KNeighborsClassifier
classifier = Pipeline([('v', CountVectorizer()),('t', TfidfTransformer()),('c',KNeighborsClassifier(60, weights='distance'))])
classifier.fit(text, text_target) 
predicted_knn = classifier.predict(valid_data)
vacc_knn = np.mean(predicted_knn == valid_target) 
result_knn = classifier.predict(test_data)


# In[33]:

met_knn = (metrics.classification_report(valid_target, predicted_knn ,digits = 2))
cmat_knn = metrics.confusion_matrix(valid_target, predicted_knn)


# In[ ]:
print("Validation Accuracy of Naive Bayes: "+vacc_nb)
print("Validation Accuracy of SVM: "+vacc_svm)
print("Validation Accuracy of K-Nearest Neighbor: "+vacc_knn)    
print("Test Dataset Result using Naive Bayes Classifier: "+result_nbc)
print("Test Dataset Result using SVM Classifier: "+result_svm)
print("Test Dataset Result using K-Nearest Neighbor Classifier: "+result_knn)
print("Metrics of Naive Bayes Classifier:"+met_nb)
print("Confusion Matrix using Naive Bayes Classifier:"+cmat_nb)
print("Metrics of SVM Classifier:"+met_svm)
print("Confusion Matrix using SVM Classifier:"+cmat_svm)
print("Metrics of K-Nearest Neighbor Classifier:"+met_knn)
print("Confusion Matrix using K-Nearest Neighbor Classifier:"+cmat_knn)


# In[ ]:



