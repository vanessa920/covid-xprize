#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df=pd.read_csv('check_this.csv')


# In[7]:


df['ratio']=df['PredictedDailyNewCases']/df['Population']


# In[14]:


pd.set_option('display.max_rows', None)
df


# In[20]:


df2.reset_index(inplace=True)


# In[21]:


df2=df[df['ratio']>0.05]
df2


# In[6]:


dfp=pd.read_csv('20201222_080002_robojudge_live.csv')

