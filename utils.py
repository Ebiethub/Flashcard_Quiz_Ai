from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests

def extract_text(source: str, is_url=False, is_pdf=False):
    """Extract text from PDF/URL/raw input"""
    if is_pdf:
        reader = PdfReader(source)
        return " ".join([page.extract_text() for page in reader.pages])
    elif is_url:
        response = requests.get(source)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    return source

def chunk_text(text, chunk_size=1000):
    """Split text for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    return splitter.split_text(text)