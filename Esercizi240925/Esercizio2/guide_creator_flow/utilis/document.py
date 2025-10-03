"""Document loading and splitting utilities.

This module provides a small helper class ``DocumentLoader`` that loads PDF
files from a directory, applies a simple PII/filtering step using the
``Checker`` utility and splits the resulting texts into chunks using
LangChain's ``RecursiveCharacterTextSplitter``.

Classes
-------
DocumentLoader
    Load PDF files from disk, filter/clean their contents and produce a
    list of LangChain ``Document`` objects. It also exposes ``split_documents``
    to convert loaded documents into text chunks suitable for embedding.
"""
import os
from typing import List
from langchain.schema import Document
from .checker import Checker
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:

    def __init__(self, directory: str, settings):
        self.directory = directory
        self.Settings=settings 
        self.load_pdfs_from_folder()

    def load_pdfs_from_folder(self) -> List[Document]:
        self.pdf_documents = []
        for filename in os.listdir(self.directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.directory, filename)
                try:
                    checker = Checker(file_path)
                    text = checker.decode(checker.pii_detection(checker.extract_pdf_text()))
                    if not checker.prompt_shield(text):
                        self.pdf_documents.append(
                            Document(page_content=text, metadata={"source": filename, "trusted": ("untrusted" not in filename)})
                        )
                    else:
                        print(f"Non è statto possibile inserire il documento {filename}")
                except Exception as e:
                    print(f"Errore nella lettura di {filename}: {e}")
        return self.pdf_documents
    
    def split_documents(self) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.Settings.chunk_size,
            chunk_overlap=self.Settings.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                "",
            ],
        )
        return splitter.split_documents(self.pdf_documents)