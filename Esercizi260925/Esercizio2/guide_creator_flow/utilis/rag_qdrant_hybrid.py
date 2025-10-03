from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

from models import get_embeddings_custom, get_llm_custom
from document import DocumentLoader
from vectore_store import VectoreStore
from search import Search

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

load_dotenv()

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
   
    

SETTINGS = Settings()



def get_embeddings():
    return get_embeddings_custom()

def get_llm():
    return get_llm_custom()
    
def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def format_docs_for_prompt(points: Iterable[Any]) -> str:
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)

def build_rag_chain(llm):
    system_prompt = (
        "Sei un assistente tecnico. Rispondi in italiano, conciso e accurato. "
        "Usa ESCLUSIVAMENTE le informazioni presenti nel CONTENUTO. "
        "Se non è presente, dichiara: 'Non è presente nel contesto fornito.' "
        "Cita sempre le fonti nel formato [source:FILE]."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "CONTENUTO:\n{context}\n\n"
         "Istruzioni:\n"
         "1) Risposta basata solo sul contenuto.\n"
         "2) Includi citazioni [source:...].\n"
         "3) Niente invenzioni.")
    ])

    chain = (
        {
            "context": RunnablePassthrough(),  # stringa già formattata
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    
    s = SETTINGS
    embeddings = get_embeddings()
    llm = get_llm()  # opzionale

    # 1) Client Qdrant
    client = get_qdrant_client(s)

    # 2) Dati -> chunk
    dl = DocumentLoader("26-Settembre/qdrant_rag_no_crewai/input_directory", Settings)
    chunks = dl.split_documents()

    # 3) Crea (o ricrea) collection
    #vector_size = embeddings._client.get_sentence_embedding_dimension()
    vs = VectoreStore(client, s, 1536, embeddings)
    vs.upsert_chunks(chunks)
    #recreate_collection_for_rag(client, s, 1536)

    # 4) Upsert chunks
    #upsert_chunks(client, s, chunks, embeddings)

    # 5) Query ibrida
    questions = [
        "Migliori hotel di Dubai",
        "Migliori hotel di New York",
        "Migliori hotel di Chicago",
    ]

    for q in questions:
        #hits = hybrid_search(client, s, q, embeddings)
        hits = Search(client, s, q, embeddings).hybrid_search()
        print("=" * 80)
        print("Q:", q)
        if not hits:
            print("Nessun risultato.")
            continue

        # Mostra id/score di debug
        for p in hits:
            print(f"- id={p.id} score={p.score:.4f} src={p.payload.get('source')}")

        # Se LLM configurato: genera
        if llm:
            try:
                ctx = format_docs_for_prompt(hits)
                chain = build_rag_chain(llm)
                answer = chain.invoke({"question": q, "context": ctx})
                print("\n", answer, "\n")
            except Exception as e:
                print(f"\nLLM generation failed: {e}")
                print("Falling back to content display...")
                print("\nContenuto recuperato:\n")
                print(format_docs_for_prompt(hits))
                print()
        else:
            # Fallback: stampa i chunk per ispezione
            print("\nContenuto recuperato:\n")
            print(format_docs_for_prompt(hits))
            print()

if __name__ == "__main__":
    main()