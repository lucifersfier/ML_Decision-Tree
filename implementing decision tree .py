#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[54]:


import warnings 
warnings.filterwarnings(action = 'ignore')
#plt.style.use(['seaborn-bright', 'dark_background'])


# # Importing dataset
# 

# In[56]:


data = pd.read_csv('churn_prediction_simple.csv')
data.head()


# Now basic Preprocessing of the data 

# In[58]:


#separating dependent and independent variables
X = data.drop(columns = ['churn','customer_id'])
Y = data['churn']


# In[59]:


#scaling the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[60]:


#Splitting the dataset 
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(scaled_X,Y,train_size = 0.80,stratify = Y)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# import the decision tree algortithm form the sklearn library 

# In[61]:


from sklearn.tree import DecisionTreeClassifier as DTC
classifier = DTC(class_weight = 'balanced')
classifier = DTC()


# we use the classifier instance to call the classifier fit function and pass the training set 'x_train & y_train' as its parameters and in the next line we will first predict the values for the training set itself and for this we use 'classifier_predict' function and pass the 'x_train' as its parameters to predict the corresponding values and store it into the 'predicted_values'

# In[63]:


classifier.fit(x_train,y_train)
predicted_values = classifier.predict(x_train)


# # performance of the decision tree model 

# print the classification report of the decision tree

# In[64]:


predicted_values[:30]


# NOW EVALUATION METRICS

# In[66]:


from sklearn.metrics import classification_report
print(classification_report(y_train,predicted_values))


# finding out the model performance over the test set 
# 

# In[68]:


predicted_values = classifier.predict(x_test)
print(classification_report(y_test,predicted_values))


# In[70]:


get_ipython().system('pip install graphviz')


# Now generating the tree structure using the classifier instance

# In[72]:


from sklearn.tree import export_graphviz #cretaes an intermediary file which contain information about the nodes and edges of any tree graph
export_graphviz(decision_tree = classifier, out_file = 'tree_viz',max_depth = None,feature_names = X.columns,label = None,impurity = False)


# In[ ]:


from graphviz import render 
render( filepath='tree_viz', format = 'png', engine = 'neato')


# physical characteristics can be tuned to prevent tree from overfitting
# preventing a decision tree from overfitting by tuning its characteristics

# In[78]:


classifier = DTC()
classifier.fit(x_train,y_train)


# # max_depth

# In[84]:


from sklearn.metrics import f1_score
def calc_score(model, x1, y1, x2, y2):
    model.fit(x1,y1)
    predict = model.predict(x1)
    f1 = f1_score(y1, predict)
    predict = model.predict(x2)
    f2 = f1_score(y2, predict)
    return f1, f2


# In[86]:


def effect(train_score,test_score, x_axis, title):
    plt.figure(figsize = (5,5), dpi = 120)
    plt.plot(x_axis, train_score, color = 'red', label = 'train_score')
    plt.plot(x_axis, test_score, color = 'blue', label = 'test_score')
    plt.title(title)
    plt.legend()
    plt.xlabel("parameter_value")
    plt.ylabel("f1_score")
    plt.show()


# In[91]:


maxdepth = [i for i in range(1,50)]
train = []
test = []
for i in maxdepth:
    model = DTC(class_weight = 'balanced',max_depth = i,random_state = 42)
    f1, f2 = calc_score(model, x_train, y_train, x_test, y_test)
    train.append(f1)
    test.append(f2)


# In[92]:


effect(train, test, range(1,50), 'max_depth')


# # min_Samples_split 

# In[94]:


min_samples = [i for i in range(2,5000,25)]efe
train = []
test = []
for i in min_samples:
    model = DTC(class_weight = 'balanced',min_samples_split = i,random_state = 42)
    f1,f2 = calc_score(model,x_train,y_train,x_test,y_test)
    train.append(f1)
    test.append(f2)


# In[ ]:


effect(train, test, range(2,5000,25), 'min_samples_split')


# # max_leaf_nodes

# In[ ]:


maxleafnodes = [i for i in range(2,200,10)]
train = []
test = []
for i in maxleafnodes:
    model = DTC(class_weight = 'balanced',max_leaf_nodes = i,random_state = 42)
    f1,f2 = calc_score(model,x_train,y_train,x_test,y_test)
    train.append(f1)
    test.append(f2)


# In[ ]:


effect(train,test,range(2,200,10),'max_leaf_nodes')


# # min_samples_leaf

# In[97]:


minsamplesleaf = [i for i in range(2,4000,25)]
train = []
test = []
for i in minsamplesleaf:
    model = DTC(class_weight = 'balanced',min_samples_leaf = i,random_state = 42)
    f1,f2 = calc_score(model,x_train,y_train,x_test,y_test)
    train.append(f1)
    test.append(f2)


# In[99]:


effect(train,test,range(2,4000,25),'min_samples_leaf')


# # feature_importance

# In[101]:


model = DTC(max_depth = 9)
model.fit(x_train,y_train)
feature_imp = pd.Series(model.feature_importances_,index=X.columns)
k = feature_imp.sort_values()


# In[103]:


plt.figure(figsize=(10,5),dpi=120)
plt.barh(k.index,k)
plt.xlabel('Importance')
plt.ylabel('Feature_name')
plt.title('feature Importance')


# In[ ]:




