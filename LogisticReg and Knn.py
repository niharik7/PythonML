#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# #### PREDICTING A PULSAR STAR using various ML models 

# In[2]:


pulsar = pd.read_csv('pulsar_stars.csv')
pulsar.head()


# Split Data using train_test_split

# In[3]:


from sklearn.model_selection import train_test_split


# X contains all the columns except the 'target_class' and y contains only the 'target_class' i.e. the actual existance of the pulsar star

# In[4]:


X = pulsar.drop('target_class',axis=1)
y = pulsar['target_class']


# train_test_split splits the data into 80% as trainging data that will be fed to the model for training and 20% as test data for which the model predicts the 'target_class'

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)


# ### Logistic Regression

# In[6]:


from sklearn.linear_model import LogisticRegression


# Train the model with X_train and y_train data

# In[7]:


logm = LogisticRegression()
logm.fit(X_train,y_train)
pred = logm.predict(X_test)


# In[8]:


print(classification_report(y_test,pred))


# In[9]:


print(confusion_matrix(y_test,pred))


# ### KNN

# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# In[11]:


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
kpred = knn.predict(X_test)


# In[12]:


print(classification_report(y_test,kpred))
print('\n')
print(confusion_matrix(y_test,kpred))


# In[13]:


error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    ipred = knn.predict(X_test)
    error.append(np.mean(ipred != y_test))


# In[14]:


plt.figure(figsize=(15,5))
plt.plot(range(1,50),error,color='orange',marker='o',markerfacecolor='purple')


# ### Decision Tree Classifier

# In[15]:


from sklearn.tree import DecisionTreeClassifier 


# In[16]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dpred = dtree.predict(X_test)


# In[17]:


print(classification_report(y_test,dpred))
print('\n')
print(confusion_matrix(y_test,dpred))


# ### Random Forest Classifier

# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train,y_train)
rfcpred = rfc.predict(X_test)


# In[20]:


print(classification_report(y_test,rfcpred))
print('\n')
print(confusion_matrix(y_test,rfcpred))

