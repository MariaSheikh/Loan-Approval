# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:32:21 2021

@author: myc
"""
import pandas as pd
import dill
import pickle
import numpy as np
import lime
import lime.lime_tabular



    
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename1='Explanation'
explaining = dill.load(open(filename1, 'rb'))
    


     
def please(testing):
    testing=testing
    result = loaded_model.predict(testing)
    
    expg = explaining.explain_instance(testing.iloc[0], loaded_model.predict_proba)
    expg.save_to_file('will.html')
    
    return result
    
    