import os
import openai
from dotenv import load_dotenv
from docx import Document
import pdfplumber

load_dotenv()


def read_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def read_doc(file_path):
    doc = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])


def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def create_embedding(text, model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY")):
    openai.api_key = api_key
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def get_text_from_file(file_path: str):
    file_ext = os.path.splitext(file_path)[1].lower()
    text = ''

    if file_ext == '.txt':
        text = read_txt(file_path)
    elif file_ext == '.doc' or file_ext == '.docx':
        text = read_doc(file_path)
    elif file_ext == '.pdf':
        text = read_pdf(file_path)
    else:
        print(f"Error: The file format {file_ext} is not supported.")
        return

    return text
