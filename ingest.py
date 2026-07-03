import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from constants import (
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    get_embeddings,
)


def file_log(logentry):
    with open("file_ingest.log", "a") as file1:
        file1.write(logentry + "\n")
    print(logentry)


def load_single_document(file_path: str):
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
        return (data_list, filepaths)


def load_documents(source_dir: str):
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = max(1, round(len(paths) / n_workers))
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            filepaths = paths[i:(i + chunksize)]
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        for future in as_completed(futures):
            try:
                contents, _ = future.result()
                docs.extend([d for d in contents if d is not None])
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs


def split_documents(documents):
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cpu",
    type=click.Choice(["cpu", "cuda", "mps"]),
    help="Device to run embeddings on. (Default is cpu)",
)
def main(device_type):
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    if not texts:
        logging.error("No valid text chunks were produced. Check SOURCE_DOCUMENTS and file_ingest.log.")
        return

    embeddings = get_embeddings()
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    db.save_local(PERSIST_DIRECTORY)
    logging.info(f"FAISS index saved to {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()