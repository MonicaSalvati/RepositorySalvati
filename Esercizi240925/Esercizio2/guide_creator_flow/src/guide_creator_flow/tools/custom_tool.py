from crewai.tools import tool
from utilis.search import Search
from utilis.vectore_store import VectoreStore
from utilis.document import DocumentLoader
 
from utilis.models import get_embeddings_custom, get_qdrant_client
from dataclasses import dataclass
 
@dataclass
class Settings:
   
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rag_chunks"
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_n_semantic: int = 30
    top_n_text: int = 100
    final_k: int = 6
    alpha: float = 0.75
    text_boost: float = 0.20
    use_mmr: bool = True
    mmr_lambda: float = 0.6
 
@tool("RAGSearch")
def RAGSearch(query: str) -> list[str]:
    """Tools module with a CrewAI tool wrapper for RAG search.

    This module exposes a single tool function ``RAGSearch`` which is registered
    as a CrewAI tool. The tool executes a hybrid retrieval pipeline using the
    project's Qdrant vector store, embeddings and simple text prefiltering.

    The implementation is intentionally procedural: it creates or recreates the
    vector collection, upserts document chunks and runs a hybrid search. The
    function returns a list of dictionary results with text, source, trust and
    id fields.

    Public API
    ----------
    Settings
        Dataclass containing default configuration parameters for the retrieval
        pipeline.

    RAGSearch(query)
        CrewAI tool function performing retrieval for the provided query.
    """
    """Run a similarity search over the vector store.
 
    Parameters
    ----------
    query : str
        The search query string.
 
    Returns
    -------
    list
        The most relevant documents for the given query, as returned by the
        underlying retriever. Each one with metadata.
 
    Examples
    --------
    >>> from guide_creator_flow.tools.custom_tool import RAGSearch
    >>> RAGSearch("Che cos'è la policy di rimborso?")  # doctest: +SKIP
    """
    client = get_qdrant_client(Settings.qdrant_url)
    embeddings = get_embeddings_custom()
 
    dl = DocumentLoader(r"C:\Users\BK476KA\OneDrive - EY\Desktop\RepositorySalvati\Esercizi240925\Esercizio2\guide_creator_flow\src\guide_creator_flow\crews\rag_crew\input_directory", Settings)
    chunks = dl.split_documents()
 
    # 3) Crea (o ricrea) collection
    #vector_size = embeddings._client.get_sentence_embedding_dimension()
    vs = VectoreStore(client, Settings, 1536, embeddings)
    vs.upsert_chunks(chunks)
 
    results = Search(client, Settings, query, embeddings).hybrid_search()

    # return [{**r.payload, "id": r.id} for r in results]
    return [{"text": r.payload["text"],"trusted": r.payload["trusted"], "source": r.payload["source"], "id": r.id} for r in results]
