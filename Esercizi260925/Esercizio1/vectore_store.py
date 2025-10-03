"""
VectoreStore module for FAISS vector store handling.

This module defines the `Settings` and `VectoreStore` classes to build,
load, and query a FAISS vector store for document embeddings. It supports
MMR and similarity search retrieval.
"""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS


# pylint: disable=too-few-public-methods
class Settings:
    """Configuration settings for vector store and retrieval.

    Attributes
    ----------
    persist_dir : str
        Directory where the FAISS index is persisted.
    chunk_size : int
        Maximum number of characters per document chunk.
    chunk_overlap : int
        Number of characters that overlap between consecutive chunks.
    search_type : str
        Type of similarity search ('mmr' or 'similarity').
    k : int
        Number of top documents to return per query.
    fetch_k : int
        Number of candidate documents to fetch before reranking (for MMR).
    mmr_lambda : float
        Trade-off parameter in MMR between relevance and diversity.
    """
    persist_dir: str = "faiss_index_example"
    chunk_size: int = 2000
    chunk_overlap: int = 400
    search_type: str = "mmr"
    k: int = 4
    fetch_k: int = 20
    mmr_lambda: float = 0.3


class VectoreStore:
    """Wrapper around a FAISS vector store for document embeddings.

    This class handles building a FAISS vector store from documents,
    loading an existing index from disk, and creating retrievers
    for similarity search or MMR search.

    Attributes
    ----------
    persist_dir : str
        Directory where the FAISS index is persisted.
    embeddings : object
        Embedding model used to vectorize documents.
    chunks : list of Document
        Document chunks to be stored in the vector store.
    vector_store : FAISS
        FAISS vector store instance.
    """

    def __init__(self, persist_dir: str, embeddings: object, chunks: List[Document]):
        """
        Initialize the vector store.

        Parameters
        ----------
        persist_dir : str
            Directory to persist or load the FAISS index.
        embeddings : object
            Embedding model used to encode documents.
        chunks : List[Document]
            List of pre-split documents to store in FAISS.

        Notes
        -----
        - If a FAISS index exists in `persist_dir`, it is loaded automatically.
        - Otherwise, a new FAISS index is built from `chunks` and saved.
        """
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.chunks = chunks

        faiss_index_path = os.path.join(Settings.persist_dir, "index.faiss")
        if os.path.exists(faiss_index_path):
            self.__load_vectorstore()
        else:
            self.__build_faiss_vectorstore()

    def __build_faiss_vectorstore(self) -> None:
        """Build a new FAISS vector store from documents and save it to disk.

        Notes
        -----
        - The directory `persist_dir` is created if it does not exist.
        - Uses `self.embeddings` to encode `self.chunks`.
        """
        self.vector_store = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(self.persist_dir)

    def __load_vectorstore(self) -> None:
        """Load an existing FAISS vector store from disk.

        Notes
        -----
        - Uses `allow_dangerous_deserialization=True` to allow loading
          custom embeddings safely.
        """
        self.vector_store = FAISS.load_local(
            Settings.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def make_retriever(self) -> object:
        """Create a retriever for querying the vector store.

        Returns
        -------
        object
            A retriever object that implements `.get_relevant_documents(query)`
            for similarity or MMR search.

        Notes
        -----
        - Uses `Settings.search_type` to select between 'mmr' and 'similarity'.
        - For MMR, `k`, `fetch_k`, and `mmr_lambda` from Settings are applied.
        """
        if Settings.search_type == "mmr":
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": Settings.k,
                    "fetch_k": Settings.fetch_k,
                    "lambda_mult": Settings.mmr_lambda,
                },
            )

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Settings.k},
        )
