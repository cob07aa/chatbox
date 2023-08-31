import os
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

# Load PDF content
pdf_path = "C:/chatbox/gbr-williams-shapps-plan-for-rail.pdf"  # Update with your PDF filename
pdf_document = fitz.open(pdf_path)

pdf_text = ""
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    pdf_text += page.get_text("text")

pdf_document.close()

# Streamlit framework
st.title('PDF Question Generator')
input_text = st.text_input("Enter a keyword to search in the PDF")

if input_text:
    st.write("Processing input...")

    # Initialize OpenAI LLM
    llm = OpenAI(temperature=0.8)

    # Create a prompt template for the questions
    question_prompt = Prompt(input_variables=["paragraph"], template="Question: {paragraph}?\nAnswer:")

    # Generate questions for relevant paragraphs
    relevant_paragraphs = [p for p in pdf_text.split("\n\n") if input_text.lower() in p.lower()]

    pdf_questions = []
    for paragraph in relevant_paragraphs:
        # Construct the prompt using the paragraph content
        paragraph_prompt = question_prompt.format(paragraph=paragraph)
        
        # Generate a question using LLMChain
        chain = LLMChain(llm=llm, prompt=paragraph_prompt, verbose=True)
        generated_question = chain.run()
        pdf_questions.append(generated_question.strip())  # Remove unnecessary whitespace

    if pdf_questions:
        st.write("Generated Questions:")
        for idx, question in enumerate(pdf_questions):
            st.write(f"{idx+1}. {question}")

    st.write("Processing complete.")


