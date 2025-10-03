import os
from dotenv import load_dotenv
from typing import List
from pathlib import Path
from langchain.schema import Document

from utilis.document import DocumentLoader
from utilis.vectore_store import VectoreStore
from utilis.models import get_embeddings

load_dotenv()


def initialize_vectorstore():
    """Initialize a FAISS vector store and return a document retriever.

    This function loads documents from a specified input directory, splits them
    into chunks, generates embeddings using the configured embedding model, and
    stores them in a FAISS vector store. It then creates a retriever object that
    can be used to query the vector store for relevant documents.

    Returns
    -------
    retriever : object
        A retriever object (typically implementing a `.get_relevant_documents(query)` method)
        that can be used to retrieve the top relevant documents for a query.

    Notes
    -----
    - The input directory is currently hardcoded in the function.
    - The vector store is created using FAISS and embeddings from `get_embeddings()`.
    - The retriever supports similarity search over document chunks.

    Examples
    --------
    >>> retriever = initialize_vectorstore()
    >>> results = retriever.get_relevant_documents("Explain RAG workflows")
    >>> len(results)
    5
    >>> results[0].page_content
    'RAG (Retrieval-Augmented Generation) combines retrieval and generation...'
    """
    input_dir = r"C:\Users\ZJ715MA\OneDrive - EY\Documents\GitHub\crewai-esercizio-nuovo\guide_creator_flow\src\guide_creator_flow\crews\rag_crew\input_directory"
    
    doc = DocumentLoader(input_dir)
    chunks = doc.split_documents()
    vector_store = VectoreStore("faiss_index_example", get_embeddings(), chunks)
    retriever = vector_store.make_retriever()
    return retriever
