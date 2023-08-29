#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore harmless warnings 

import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Import psql to run queries 

import pandasql as psql


# In[2]:


#2 Load the dataset

train = pd.read_csv(r"C:\Users\aparn\Downloads\20b91a5709-proj\Train.csv", header=0)


# Copy the file to back-up file

train_bk = train .copy()

# Display first 5 records

train .head()


# In[3]:


#3 Display the data information

train.info()


# In[4]:


train.shape


# In[5]:


# Display the unique values of all the variables

train.nunique()


# In[6]:


#4 Displaying Duplicate values with in Loan ataset, if avialble

train_dup = train[train.duplicated(keep='last')]
train_dup


# In[7]:


# Count the missing values by each variable, if available

train.isnull().sum()


# In[8]:


train['VAR38'].value_counts()


# In[9]:


train.drop(['VAR37','VAR33','VAR23','VAR8','VAR4','VAR2','VAR3','VAR28','VAR11','VAR20','VAR34',],axis=1,inplace=True)


# In[10]:


train.columns


# In[11]:


train.shape


# In[12]:


train.isnull().sum()


# In[13]:


from sklearn.impute import KNNImputer

imputer_int = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean',
                         copy=True, add_indicator=False)

train['VAR5'] = imputer_int.fit_transform(train[['VAR5']])
train['VAR6'] = imputer_int.fit_transform(train[['VAR6']])
train['VAR9'] = imputer_int.fit_transform(train[['VAR9']])
train['VAR14'] = imputer_int.fit_transform(train[['VAR14']])
train['VAR17'] = imputer_int.fit_transform(train[['VAR17']])
#train['VAR30'] = imputer_int.fit_transform(train[['VAR30']])
#train['VAR35'] = imputer_int.fit_transform(train[['VAR35']])


# In[14]:


# Identify the numerical and categorical variables

num_vars = train.columns[train.dtypes != 'object']
cat_vars = train.columns[train.dtypes == 'object']
print(num_vars)
print(cat_vars)


# In[15]:


train.isnull().sum()


# In[16]:


# Use LabelEncoder to handle categorical data

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

train['VAR30'] = LE.fit_transform(train[['VAR30']])
train['VAR35'] = LE.fit_transform(train[['VAR35']])
train['VAR10'] = LE.fit_transform(train[['VAR10']])
train['VAR12'] = LE.fit_transform(train[['VAR12']])
train['VAR13'] = LE.fit_transform(train[['VAR13']])
train['VAR15'] = LE.fit_transform(train[['VAR15']])
train['VAR16'] = LE.fit_transform(train[['VAR16']])
train['VAR18'] = LE.fit_transform(train[['VAR18']])
train['VAR19'] = LE.fit_transform(train[['VAR19']])
train['VAR21'] = LE.fit_transform(train[['VAR21']])
train['VAR22'] = LE.fit_transform(train[['VAR22']])
train['VAR24'] = LE.fit_transform(train[['VAR24']])
train['VAR26'] = LE.fit_transform(train[['VAR26']])


# In[17]:


train['VAR7'] = LE.fit_transform(train[['VAR7']])
train['VAR25'] = LE.fit_transform(train[['VAR25']])
train['VAR27'] = LE.fit_transform(train[['VAR27']])


# In[18]:


train.isnull().sum()


# In[19]:


# Identify the independent and Target (dependent) variables

IndepVar = []
for col in train.columns:
    if col != 'VAR38':
        IndepVar.append(col)

TargetVar = 'VAR38'

x =train[IndepVar]
y = train[TargetVar]


# In[20]:


# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)

# Display the shape of train and test data 

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[21]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train = mmscaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = mmscaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)


# In[25]:


# Load the Results dataset

CSResults = pd.read_csv(r"C:\Users\aparn\Downloads\resultfileknnalgorithmswithknnresultfile\HTResults.csv", header=0)

CSResults.head()


# In[26]:


# To build the 'Random Forest' model with random sampling

from sklearn.ensemble import RandomForestClassifier

# Create model object

ModelRF = RandomForestClassifier()
#ModelRF = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
#                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
#                                 max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, 
#                                 n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, 
#                                 ccp_alpha=0.0, max_samples=None)

# Train the model with train data 

ModelRF.fit(x_train,y_train)

# Predict the model with test data set

y_pred = ModelRF.predict(x_test)
y_pred_prob = ModelRF.predict_proba(x_test)

# Confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3);
specificity = round(tn/(tn+fp), 3);
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
balanced_accuracy = round((sensitivity+specificity)/2, 3);
precision = round(tp/(tp+fp), 3);
f1Score = round((2*tp/(2*tp + fp + fn)), 3);

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)

# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, ModelRF.predict_proba(x_test)[:,1])
plt.figure()
#--------------------------------------------------------------------
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show() 
print('-----------------------------------------------------------------------------------------------------')


# In[29]:


# Build the Calssification models with Over Sampling and compare the results

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Create objects of classification algorithms with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()

# Evalution matrix for all the algorithm

MM = [ModelLR, ModelDC, ModelRF, ModelET]
for models in MM:
            
    # Train the model training dataset
    
    models.fit(x_train, y_train)
    
    # Prediction the model with test dataset
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    #from math import sqrt

    #mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    #MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    #print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    Model_roc_auc = roc_auc_score(actual, predicted)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    #
    plt.plot(fpr, tpr, label= 'Classification Model' % Model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': 'MCC',
               'ROC_AUC_Score':roc_auc_score(actual, predicted),
               'Balanced Accuracy':balanced_accuracy}
    CSResults = CSResults.append(new_row, ignore_index=True)
    #----------------------------------------------------------------------------------------------------------


# In[30]:


CSResults.head()


# In[31]:


# Hyperparameter tuning using RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # number of trees in the random forest 
max_features = ['auto', 'sqrt', 'log2'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap}


# In[ ]:


# Importing Random Forest Classifier from the sklearn.ensemble

#from sklearn.ensemble import RandomForestClassifier

#ModelRF4 = RandomForestClassifier()

#from sklearn.model_selection import RandomizedSearchCV

#rf_random = RandomizedSearchCV(estimator = ModelRF4,param_distributions = random_grid,
                               #n_iter = 100, cv = 3, verbose=2, random_state=35, n_jobs = -1)

# Fit the model with train data

#rf_random.fit(x_train, y_train)


# In[ ]:


# Print the best parameters

#print ('Random grid: ', random_grid, '\n')

# print the best parameters

#print ('Best Parameters: ', rf_random.best_params_, ' \n')

