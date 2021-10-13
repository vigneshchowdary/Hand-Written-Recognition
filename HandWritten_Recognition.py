#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# In[3]:


digits=load_digits()


# In[37]:


digits


# In[38]:


plt.gray()

for i in range(10):
    plt.matshow(digits.images[i])
    


# In[6]:


dir(digits)


# In[15]:


digits.DESCR


# In[17]:


digits.frame


# In[7]:


digits.target_names


# In[10]:


digits.target


# In[12]:


digits.target.shape


# In[13]:


digits.data


# In[14]:


digits.feature_names


# In[19]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


xtrain,xtest,ytrain,ytest=train_test_split(digits.data, digits.target,test_size=0.2)


# In[22]:


ytrain.shape


# In[23]:


xtrain.shape


# In[24]:


model.fit(xtrain,ytrain)


# model.score(xtest,ytest)

# In[27]:


model.score(xtest,ytest)


# In[30]:


model_predicted=model.predict(xtest)


# In[31]:


model_predicted


# In[32]:


ytest


# In[33]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,model_predicted)
cm


# In[46]:


import seaborn as sn
plt.figure(figsize=(12,6))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




