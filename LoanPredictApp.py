# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:56:21 2021

@author: myc
"""
import pandas as pd
import streamlit as st
import pandas as pd
import dill
import pickle
import numpy as np
import lime
import lime.lime_tabular

st.write("""
# Loan Approval Prediction App
This app predicts whether loan would be approved or not and features affecting the decision
""")


def user_input_features():
    a=st.text_input('Enter age')
    b=st.text_input('Enter Income')
    c=st.text_input('Enter Credit Score')
    d=st.text_input('Enter House hold size')
    e=st.text_input('Enter Median Home Value')
    f=st.text_input('Enter Median Household Income')
    g=st.text_input('Enter Debt')
    h=st.text_input('Enter Loan Term')
    i=st.text_input('Enter Interest Rate')
    j=st.text_input('Enter Credit Incidents')
    k=st.text_input('Enter HomeValue')
    l=st.text_input('Enter Loan Amount')
    m=st.selectbox('Select Loan type', ['Government_insured','Fixed_rate','Adjustable_rate','Jumbo'])
    st.text("")
    st.text("")
    st.text("")
    n=1   
    if(m=='Government_insured'):
        n=1
    elif(m=='Fixed_rate'):
        n=2
    elif(m=='Adjustable_rate'):
        n=3
    elif(m=='Jumbo'):
        n=4   
    data = {'Age': int(a),
                'Income': int(b),
                'CreditScore': int(c),
                'HouseholdSize': int(d),
                'MedianHomeValue': int(e),
                'MedianHouseholdIncome': int(f),
                'Debt': float(g),
                'LoanTerm': int(h),
                'InterestRate': float(i),
                'CreditIncidents': int(j),
                'HomeValue': int(k),
                'LoanAmount': int(l),
                'ProductType': int(n)     
                }
    features = pd.DataFrame(data, index=[0])
    return features


def please(testing):
    testing=testing
    result = loaded_model.predict(testing)
    expg = explaining.explain_instance(testing.iloc[0], loaded_model.predict_proba)
    expg.save_to_file('will.html')
    return result

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename1='Explanation'
explaining = dill.load(open(filename1, 'rb'))




testing = user_input_features()
ma=st.button('Predict!!')
st.text("")
st.text("")
if ma:
    m1=please(testing)
    if(m1==1):
        st.write('Hurrah! There are strong chances that your loan would be accepted.')
    elif(m1==0):
        st.write('Sorry! There are not any bright chance of getting your loan accepted')
    st.text("")
    st.text("")
    st.write('Below is the description about features affecting the decision')
    st.write('1 depicts loan approaval and 0 depicts disapproval')
    
    st.text("")
    st.text("")
    st.text("")
    import streamlit.components.v1 as components
    HtmlFile = open("will.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code, height=1000)