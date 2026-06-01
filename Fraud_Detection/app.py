import pandas as pd
import numpy as np
import joblib
import requests
import streamlit as st 

model= joblib.load('./Fraud_Detection/fraud_detection_pipeline.pkl')

st.title('Fraud Detection Prediction App')

st.markdown("Please enter the transaction details and use predict button")

st.divider()


transaction_type=st.selectbox('Transaction Type' ,options=(['PAYMENT','TRANSFER','CASH_OUT','DEPOSIT']))

amount=st.number_input("Amount",min_value=0.0, value=1000.0,step=100.0)

oldbalanceOrg=st.number_input('Old Balance (Sender)', min_value=0.0, value=10000.0)
newbalanceOrig=st.number_input("New Balance (Sender)",min_value=0.0, value=9000.0)


oldbalanceDest=st.number_input('Old Balance (Recivier)', min_value=0.0, value=0.0)
newbalanceDest=st.number_input("New Balance (Recivier)",min_value=0.0, value=0.0)


if st.button("Predict"):
    input_data=pd.DataFrame([{
        'type':transaction_type,
        'amount':amount,
        'oldbalanceOrg':oldbalanceOrg,
        'newbalanceOrig':newbalanceOrig,
        'oldbalanceDest':oldbalanceDest,
        'newbalanceDest':newbalanceDest
        
        
    }])
    
    prediction=model.predict(input_data)[0]

    st.subheader(f"Prediction: '{int(prediction)}' ")

    if prediction==1:
        st.error('This transaction can be fraud')
        
    else:
        st.success('This transaction looks like it is not a fraud')















