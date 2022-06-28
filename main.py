import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("My First Streamlit Demo")

form = st.form(key='my_form')
textinput = form.text_input(label='Enter some text')
submit_button = form.form_submit_button(label='Submit')

loaded_model = joblib.load('LR_model.joblib')

sentiment = loaded_model.predict([textinput])[0]

if submit_button:
    st.write(f'{sentiment}')