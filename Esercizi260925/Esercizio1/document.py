import os
from typing import List
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Settings:
    """Configuration settings for document chunking and vector store retrieval.

    Attributes
    ----------
    persist_dir : str
        Directory where the vector store index is persisted.
    chunk_size : int
        Maximum number of characters per document chunk.
    chunk_overlap : int
        Number of characters that overlap between consecutive chunks.
    search_type : str
        Type of similarity search to use (e.g., 'mmr' for Maximal Marginal Relevance).
    k : int
        Number of top documents to return for a query.
    fetch_k : int
        Number of candidate documents to fetch before reranking.
    mmr_lambda : float
        Parameter controlling the trade-off in MMR between relevance and diversity.
    """
    #persist_dir: str = "faiss_index_example"
    chunk_size: int = 2000
    chunk_overlap: int = 400
    #search_type: str = "mmr"
    #k: int = 4
    #fetch_k: int = 20
    #mmr_lambda: float = 0.3


class DocumentLoader:
    """Load PDF documents from a folder and split them into chunks.

    This class handles reading PDFs, extracting text, and splitting content
    into smaller chunks suitable for vectorization and retrieval.

    Attributes
    ----------
    directory : str
        Path to the folder containing PDF files.
    pdf_documents : list of Document
        List of loaded PDF documents as LangChain Document objects.
    """

    def __init__(self, directory: str):
        """
        Initialize the DocumentLoader.

        Parameters
        ----------
        directory : str
            Path to the folder containing PDF files.
        """
        self.directory = directory
        self.load_pdfs_from_folder()

    def load_pdfs_from_folder(self) -> List[Document]:
        """Load all PDF files from the specified directory.

        Each PDF is converted to a `Document` object with `page_content`
        containing the extracted text and `metadata` containing the filename.

        Returns
        -------
        List[Document]
            A list of `Document` objects representing all PDFs in the directory.

        Notes
        -----
        - PDFs that cannot be read will be skipped, with an error printed to
          standard output.
        """
        self.pdf_documents = []
        for filename in os.listdir(self.directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.directory, filename)
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    self.pdf_documents.append(
                        Document(page_content=text, metadata={"source": filename})
                    )
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        return self.pdf_documents

    def split_documents(self) -> List[Document]:
        """Split loaded PDF documents into smaller chunks.

        Uses `RecursiveCharacterTextSplitter` with settings defined in the
        `Settings` class. Splits text based on a hierarchy of separators to
        create overlapping chunks suitable for embedding and retrieval.

        Returns
        -------
        List[Document]
            A list of chunked `Document` objects.

        Notes
        -----
        - Each chunk contains up to `Settings.chunk_size` characters.
        - Consecutive chunks overlap by `Settings.chunk_overlap` characters.
        - Splitting uses separators in the following order: paragraph, line,
          sentence-ending punctuation, space, and empty string.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap,
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
