#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn  import preprocessing


# # Load DATA from CSV file 

# In[33]:


data = pd.read_csv('teleCust1000t.csv')
data.head(-3)


# # Data Visualization and Analysis

# In[34]:


# how many of each class is in our data set
data['custcat'].value_counts()


# In[35]:


data.hist(column ='income', bins = 50)


# In[36]:


# Feature Set 

X = data[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
y = data[['custcat']].values


# In[37]:


X[0:3]


# # Normalize Data 

# In[40]:


#Data Standardization 

Scaler = StandardScaler()
X = Scaler.fit_transform(X)
X[0:2]


# In[41]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Model  K nearest neighbor (KNN)

# In[47]:


from sklearn.neighbors  import KNeighborsClassifier


# In[53]:


# Training 
# for K = 4 
K = 4
Model = KNeighborsClassifier(n_neighbors = K)
Model.fit(X_train,y_train)
y_hat = Model.predict(X_test)
y_hat[0:5]


# In[55]:


#Evaluation 
from sklearn import metrics 
print('test set accuracy :', metrics.accuracy_score(y_test,y_hat))


# # Best Model

# In[70]:


from sklearn.model_selection import GridSearchCV
param = {'n_neighbors': np.arange(1,100),
        'metric' : ['eucludean', 'manhattan']}


# In[71]:


GRID = GridSearchCV(KNeighborsClassifier() ,param , cv =5)


# In[72]:


GRID.fit(X_train , y_train)


# In[73]:


GRID.best_score_


# In[74]:


GRID.best_params_


# In[75]:


GRID.best_estimator_


# In[76]:


model1=GRID.best_estimator_
model1.score(X_test,y_test)


# In[ ]:




