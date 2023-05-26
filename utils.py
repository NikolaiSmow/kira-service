def get_number_of_words(content: str):
    if content:
        return len(content.split(" ")) - 1
    else:
        return 0


def get_number_of_characters(content: str):
    if content:
        return len(content)
    else:
        return 0


def convert_lines_to_characters(num_lines: int, num_characters_per_line: int):
    return num_lines * num_characters_per_line


def convert_characters_to_lines(num_characters: int, num_characters_per_line: int):
    return num_characters // num_characters_per_line


import re
from io import BytesIO
from typing import Any, Dict, List

# import docx2txt
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from openai.error import AuthenticationError

# from pypdf import PdfReader

# from knowledge_gpt.embeddings import OpenAIEmbeddings
# from knowledge_gpt.prompts import STUFF_PROMPT


# @st.experimental_memo()
# def parse_docx(file: BytesIO) -> str:
#     text = docx2txt.process(file)
#     # Remove multiple newlines
#     text = re.sub(r"\n\s*\n", "\n\n", text)
#     return text


# @st.experimental_memo()
# def parse_pdf(file: BytesIO) -> List[str]:
#     pdf = PdfReader(file)
#     output = []
#     for page in pdf.pages:
#         text = page.extract_text()
#         # Merge hyphenated words
#         text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
#         # Fix newlines in the middle of sentences
#         text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
#         # Remove multiple newlines
#         text = re.sub(r"\n\s*\n", "\n\n", text)
#         output.append(text)

#     return output


# @st.experimental_memo()
# def parse_txt(file: BytesIO) -> str:
#     text = file.read().decode("utf-8")
#     # Remove multiple newlines
#     text = re.sub(r"\n\s*\n", "\n\n", text)
#     return text
