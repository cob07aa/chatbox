## Integrate code Transformers 

import streamlit as st
from transformers import pipeline

# streamlit framework

st.title('Langchain Demo With Hugging Face Transformers')
input_text = st.text_input("Search the topic you want")

# Create a conversational pipeline using Hugging Face's Transformers
conversational_pipeline = pipeline("conversational")

if input_text:
    # Generate a response using the conversational pipeline
    response = conversational_pipeline(input_text)[0]["generated_text"]
    st.write(response)
