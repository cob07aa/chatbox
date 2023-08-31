import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from constants import openai_key
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain import PromptTemplate
from langchain.prompts.prompt import Prompt  # Import the Prompt class
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chains import SequentialChain


os.environ["OPENAI_API_KEY"] = openai_key


# Streamlit framework
st.title('Langchain Demo With OpenAI API')
input_text = st.text_input("Search the topic you want")

# Initialize OpenAI LLMS
llm = OpenAI(temperature=0.8)

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

# Create OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Create a Vectorstore index using FAISS from the texts and embeddings
document_search = FAISS.from_texts(texts, embeddings)

if input_text:
    # Use LLMS to generate text
    st.write(llm(input_text))
    
    st.write("Searching similar content in PDF...")
    
    # Perform similarity search using document_search
    docs = document_search.similarity_search(input_text)
    
    # Load QA Chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
    # Run QA Chain with input documents and question
    response = chain.run(input_documents=docs, question=input_text)
    st.write("Response from QA Chain:")
    st.write(response)

    # Query the index

    # Online PDF Loader
    loader = OnlinePDFLoader("C:/chatbox/gbr-williams-shapps-plan-for-rail.pdf")  # Replace with your PDF file path
    data = loader.load()

    # Create Vectorstore index from loader
    index = VectorstoreIndexCreator().from_loaders([loader])
    index_query_result = index.query(input_text)
    st.write("Results from Vectorstore Index Query:")
    st.write(index_query_result)
