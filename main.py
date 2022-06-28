import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("My First Streamlit Demo")
st.markdown('Sentiment Analysis. Logistic Regression model trained using COVID-19 vaccines English tweets.')

form = st.form(key='my_form')
textinput = form.text_input(label='Enter some text')
submit_button = form.form_submit_button(label='Submit')

loaded_model = joblib.load('LR_model.joblib')

sentiment = loaded_model.predict([textinput])[0]

if submit_button:
	if sentiment=='positive':
		st.success('Positive Sentiment!')
	elif sentiment=='negative':
		st.error('Negative Sentiment!')
	elif sentiment=='neutral':
		st.info('Neutral Sentiment!')


    