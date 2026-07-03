import os
from functools import partial

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    PDFMinerLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings

# Root
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Database
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS")

# Persist Directory
PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "DB")

MODELS_PATH = os.path.join(ROOT_DIRECTORY, "models")

INGEST_THREADS = os.cpu_count() or 8

# Context window / generation length for the LLM
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = 150

# File extension -> loader class
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": partial(TextLoader, autodetect_encoding=True),
    ".md": UnstructuredMarkdownLoader,
    ".py": partial(TextLoader, autodetect_encoding=True),
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """Returns the HuggingFace embedding function used for both ingest.py and run_ChatIITK.py."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"