# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:23:56 2021

@author: myc
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import pickle


#simplefilter(action='ignore', category=FutureWarning)
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('Trail.csv')

v1=df.dropna()
y = v1['LoanStatus']
X = v1.drop(['LoanStatus'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=.20, random_state=0)
undersample = RandomUnderSampler(sampling_strategy='majority')
X_over, y_over = undersample.fit_resample(x_train, y_train)
import pickle
clf3 = RandomForestClassifier(n_estimators= 100)
clf3.fit(X_over, y_over)
filename = 'finalized_model.sav'
pickle.dump(clf3, open(filename, 'wb'))


import lime
import lime.lime_tabular
X_featurenames = X_over.columns
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_over),
                    feature_names=X_featurenames, class_names=['0','1'], discretize_continuous=True)
import dill
#pickle.dump(explainer, open(filename1, 'wb'))
with open('Explanation', 'wb') as f:
    dill.dump(explainer, f)

            
            