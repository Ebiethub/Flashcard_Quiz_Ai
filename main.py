import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils import extract_text, chunk_text

load_dotenv()

# Initialize Groq model
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-specdec",
        temperature=0.7,
        max_tokens=4000
    )

# ===== Core Functionalities =====
def generate_flashcards(text):
    """Generate flashcards using Groq"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Create 5 flashcards from this text. Use this format:
        Question: [question]
        Answer: [answer]
        ---
        {text}
        """
    )
    chain = LLMChain(llm=groq_chat, prompt=prompt)
    return chain.run(text)

def generate_quiz(text):
    """Generate quiz questions"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Create a 3-question quiz with MCQ and true/false questions.
        Include answers. Text: {text}
        """
    )
    chain = LLMChain(llm=groq_chat, prompt=prompt)
    return chain.run(text)

    
# ===== Streamlit UI =====
st.title("AI Learning Assistant 🤖")

# Input options
input_type = st.radio("Choose input type:", ["Text", "PDF", "URL"])
content = ""

if input_type == "Text":
    content = st.text_area("Paste your content:")
elif input_type == "PDF":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        content = extract_text(uploaded_file, is_pdf=True)
elif input_type == "URL":
    url = st.text_input("Enter URL:")
    if url:
        content = extract_text(url, is_url=True)

# Process content
if content:
    chunks = chunk_text(content)
    
    if st.button("Generate Flashcards"):
        with st.spinner("Creating flashcards..."):
            result = generate_flashcards(chunks[0])
            st.markdown("### Flashcards")
            st.write(result.replace("---", "\n"))

    if st.button("Generate Quiz"):
        with st.spinner("Building quiz..."):
            result = generate_quiz(chunks[0])
            st.markdown("### Quiz")
            st.write(result)
