import os
from constants import openai_key
from openai import Completion
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
import streamlit as st
from openai import OpenAI


# Set your OpenAI API key here

import os
import streamlit as st
from openai import Completion
from PyPDF2 import PdfReader

openai_api_key = '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'

# Streamlit framework
st.title('RSSB Chatbot Demo With OpenAI API')
input_text = st.text_input("Search the topic you want")

# Initialize OpenAI API
temperature = 0.8

# Load PDF content using PdfReader
pdf_path = 'C:/chatbox/gbr-williams-shapps-plan-for-rail.pdf'
pdfreader = PdfReader(pdf_path)

raw_text = '\n'.join(page.extract_text() for page in pdfreader.pages if page.extract_text())

if input_text:
    # Use OpenAI to generate text
    llm_response = Completion.create(
        engine="davinci-codex",
        prompt=input_text,
        temperature=temperature,
        max_tokens=100,
        api_key=openai_api_key
    )
    generated_text = llm_response.choices[0].text
    st.write("Generated Text:")
    st.write(generated_text)
    
    st.write("Searching similar content in PDF...")
    
    # Process the raw text into chunks
    chunk_size = 800
    chunk_overlap = 200
    texts = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size - chunk_overlap)]

    # Get the most relevant text chunk based on the input
    closest_text = max(texts, key=lambda text: text.count(input_text))
    
    st.write("Most relevant chunk of text:")
    st.write(closest_text)
