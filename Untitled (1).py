#!/usr/bin/env python
# coding: utf-8

# Problem Statement Scenario:
# 
# Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
# 
# To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.
# 
# You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.
# 
# Tasks to be performed:
# 
# If for any column(s), the variance is equal to zero, then you need to remove those variable(s). Check for null and unique values for test and train sets. Apply label encoder. Perform dimensionality reduction. Predict your test_df values using XGBoost.

# In[1]:


## Import the libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Import datasets 

train_data = pd.read_csv("train.csv") # train dataset
test_data = pd.read_csv("test.csv") # test dataset


# # Exploring data

# In[3]:


## first gonna start with test data

test_data.head(10)


# In[4]:


test_data.isnull() ## we checking for any missing values 


# In[5]:


test_data.shape


# As we can see the shape of test data has 4209 columns and 377 rows

# In[6]:


test_data.describe() ## statistical feeling of test data


# In[7]:


## My focus is on the train dataset

train_data.head(20)


# In[8]:


## gonna check for missing values 

train_data.isnull().sum()


# In[9]:


## They is no missing that in the train data


# In[10]:


train_data['y'].isnull().sum()


# In[11]:


train_data.shape


# The data has 4209 columns and 378 rows

# In[12]:


train_data.describe()


# The above table shows the statistics of train_data as a whole

# In[13]:


train_data['y'].describe()


# This is when we look at the y column in the train data we can see that the data type is a float

# # Distribution on target values

# In[14]:


import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

plt.figure(figsize = (10, 5)) 
plt.subplot()
sns.distplot(train_data.y.values, bins = 15, color = 'red')

plt.title('Distribution target in values \n', fontsize = 10)
plt.xlabel('Target value in seconds')
plt.ylabel('occurances')
plt.subplot()


# The distribution is positively skewed where mean is at 100, to be accurate I will do boxplot graph to see outliers within the data.

# In[15]:


sns.boxplot(train_data.y.values, color  = 'blue')
plt.title('Distribution in target values \n', fontsize = 10)
plt.xlabel('Target value in seconds')
plt.subplot()


# The above boxplot show outliers from 130 and above, with one outlier around 260. 

# In[16]:


## we gonna look at how target values frequently changes using time serie

plt.figure(figsize = (20, 10))
plt.plot(train_data.y.values, color = 'green')
plt.xlabel("ID")
plt.ylabel('Target')
plt.title("The change in target/ frequency movement of target")
plt.show()


# # More analysis

# In[17]:


train_data.dtypes[train_data.dtypes == 'object']


# In[18]:


obj_dtype = train_data.dtypes[train_data.dtypes=='object'].index
for i in obj_dtype:
    print(i, train_data[i].unique())


# In[19]:


fig,ax = plt.subplots(len(obj_dtype), figsize=(20,80))

for i, col in enumerate(obj_dtype):
    sns.boxplot(x=col, y='y', data=train_data, ax=ax[i])


# In[20]:


df = train_data.dtypes[train_data.dtypes=='int'].index[1:]


# In[21]:


df


# In[22]:


nan_num = []
for i in df:
    if (train_data[i].var()==0):
        print(i, train[i].var())
        nan_num.append(i)


# We have a set of numeric variables, where the value is set to 1 or 0, so there is no need to carry out volumetric analysis.

# # XGBoost

# In[23]:


columns = list(set(train_data.columns) - set(['ID', 'y']))
y_train = train_data['y'].values
ID_test = test_data['ID'].values
x_train = train_data[columns]
x_test = test_data[columns]


# In[24]:


for column in columns:
    cardi = len(np.unique(x_train[column]))
    if cardi == 1:
        x_train.drop(column, axis = 1)
        x_test.drop(column, axis = 1)
     
    if cardi > 2:
        mapp = lambda x: sum([ord(digit) for digit in x])
        x_train[column] = x_train[column].apply(mapp)
        x_test[column] = x_test[column].apply(mapp)
x_train.head()


# In[28]:


get_ipython().system('pip install xgboost')


# In[29]:


import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=10)
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(x_test)
params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score
, maximize=True, verbose_eval=10)


# In[35]:


p_test = clf.predict(d_test)

df = pd.DataFrame()
df['ID'] = ID_test
df['y'] = p_test
df.to_csv('xgb_results.csv',index=False)


# In[36]:


df.head()


# In[ ]:




