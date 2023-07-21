#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Mall_Customers.csv')
df.head()


# # Univariate Analysis

# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


df.info()


# In[48]:


from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


df.corr()


# In[6]:


sns.distplot(df['Annual Income (k$)'] );


# In[7]:


df.columns


# In[8]:


col = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in col:
    plt.figure()
    sns.distplot(df[i])


# In[9]:


df = df.rename(columns={'Genre':'Gender'})


# In[10]:


df1 = pd.melt(df , id_vars=['Gender'] , value_vars=['Annual Income (k$)'] , var_name = 'Metric')
sns.kdeplot(data=df1 , x = 'value' , shade = True , hue='Gender' );


# In[11]:


df


# In[12]:


col = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in col:
    plt.figure()
    sns.kdeplot(data=df , x=i , shade=True , hue='Gender')


# In[13]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[14]:


sns.scatterplot(data=df , x='Annual Income (k$)' ,  y= 'Spending Score (1-100)')


# In[15]:


df= df.drop('CustomerID' , axis=1)


# In[16]:


sns.pairplot(df , hue='Gender');


# In[17]:


df.groupby(['Gender'])['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# In[18]:


df.corr()


# In[19]:


sns.heatmap(df.corr() , cmap='coolwarm' ,annot=True);


# # Clustering - Univariate, Bivariate, Multivariate

# In[20]:


clustering1 = KMeans(n_clusters=3)


# In[21]:


clustering1.fit(df[['Annual Income (k$)']])


# In[22]:


clustering1.labels_


# In[23]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[24]:


df['Income Cluster'].value_counts()


# In[25]:


clustering1.inertia_


# In[26]:


inertia_scores =[]
for i in range(1 , 11):
    kmean = KMeans(n_clusters = i)
    kmean.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmean.inertia_)


# In[27]:


inertia_scores


# In[28]:


plt.plot(range(1,11) , inertia_scores);


# In[29]:


df.columns


# In[30]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# # Bivariate Clustering

# In[31]:


clustering2=KMeans(n_clusters=5)
clustering2.fit(df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[32]:


inertia_scores2 =[]
for i in range(1 , 11):
    kmean2 = KMeans(n_clusters = i)
    kmean2.fit(df[['Annual Income (k$)' , 'Spending Score (1-100)']])
    inertia_scores2.append(kmean2.inertia_)
plt.plot(range(1,11) , inertia_scores2);


# In[33]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y' , 0]


# In[34]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[35]:


pd.crosstab(df['Spending and Income Cluster'] , df['Gender'] , normalize = 'index')


# In[36]:


df.groupby('Spending and Income Cluster')['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# In[37]:


#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[38]:


scale = StandardScaler()


# In[39]:


df.head()


# In[40]:


dff = pd.get_dummies(df , drop_first =True)
dff.head()


# In[41]:


dff.columns


# In[42]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff


# In[43]:


dff = scale.fit_transform(dff)


# In[44]:


df = pd.DataFrame(scale.fit_transform(dff))


# In[45]:


inertia_scores3 =[]
for i in range(1 , 11):
    kmean3 = KMeans(n_clusters = i)
    kmean3.fit(df)
    inertia_scores3.append(kmean3.inertia_)
plt.plot(range(1,11) ,inertia_scores3 )


# In[46]:


df


# In[ ]:


df


# In[ ]:




