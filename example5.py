import os
from constants import openai_key
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
import faiss

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key


# Load PDF content using PdfReader
pdfreader = PdfReader('C:/chatbox/gbr-williams-shapps-plan-for-rail.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Split the raw text into chunks using a Text Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Load Universal Sentence Encoder from TensorFlow Hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Generate embeddings for text chunks using Universal Sentence Encoder
text_embeddings = np.array(embed(texts))

# Create a TextLoader for Langchain
text_loader = TextLoader(texts)

# Create a FAISS index
index = faiss.IndexFlatL2(text_embeddings.shape[1])
index.add(text_embeddings)

st.title('Langchain Demo With OpenAI API')
input_text = st.text_input("Search the topic you want")

if input_text:
    st.write("Searching similar content in PDF...")
    
    # Convert input text into an embedding
    input_embedding = np.array(embed([input_text]))
    
    # Perform similarity search using FAISS index
    _, similar_indices = index.search(input_embedding, k=len(texts))
    
    st.write("Similar Texts:")
    for idx in similar_indices[0]:
        st.write(texts[idx])
