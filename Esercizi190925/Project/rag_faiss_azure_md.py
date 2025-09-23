from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
from langchain.schema import Document

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ddgs import DDGS

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Vector store (FAISS)
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

# --- RAGAS ---
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

# =========================
# Configurazione
# =========================

load_dotenv()
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
azure_endpoint = os.getenv("ENDPOINT")
model_name = os.getenv("MODEL_NAME")

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 400
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.7         # 0 = diversificazione massima, 1 = pertinenza massima
    web_keyword : str = "Francia"



SETTINGS = Settings()

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset

# =========================
# Componenti di base
# =========================

def get_embeddings() -> AzureOpenAIEmbeddings:
    """
    Restituisce un modello di embedding di azure.
    """
    return AzureOpenAIEmbeddings(
        model=os.getenv("EMBEDDING_NAME"),
        azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
        api_version=os.getenv("EMBEDDING_API_VERSION"),
    )


def get_llm_from_azure():
    """
    Restituisce il modello GPT (chat) deployato su Azure OpenAI.
    """
    return AzureChatOpenAI(
        model=os.getenv("GPT_NAME"),
        azure_endpoint=os.getenv("GPT_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
        api_version=os.getenv("GPT_API_VERSION"),
        temperature=0
    )


def load_and_split_documents(folder_path: str, settings: Settings):
    """
    Carica e splitta documenti da una cartella in base al tipo di file.
    file_type può essere 'pdf' o 'txt'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    loader = DirectoryLoader(
        folder_path,
        loader_cls=TextLoader
    )
    return loader.load_and_split(splitter)


def build_faiss_vectorstore(chunks: List[Document], embeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings, chunks) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in inglese."
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)

def read_questions(file_path: str):
    """
    Legge un file di testo e restituisce una lista di domande.
    Ogni domanda deve essere separata da due newline (\n\n).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split su due newline consecutivi
    questions = [q.strip() for q in content.split("\n\n") if q.strip()]
    return questions

# =========================
# Esecuzione dimostrativa
# =========================

def main():
    settings = SETTINGS

    # 1) Componenti
    embeddings = get_embeddings()
    llm = get_llm_from_azure()

    # 2) Dati e indice FAISS
    docs = load_and_split_documents("Documents", settings)
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)



# 5) Esempi di domande
    # questions = read_questions("questions.txt")
    questions = ["Tell me the author of the project","Which performance metrics have been used?","Has the system been evaluated?"]
    

    # (opzionale) ground truth sintetica per correctness
    ground_truth = {
        questions[0]: "The author of the project is Alessio Buda.",
        questions[1]: "The performance metrics used are:security accuracy rate, threat detection success rate, false positive/negative rates, and response time per evaluation.\n\n",
        questions[2]: "The system has been evaluated through a comprehensive testing and evaluation framework that validates the performance, security, and quality of all three crews. The testing system uses MLflow for experiment tracking and provides both automated and manual testing capabilities. The evaluation framework consists of three main modules: Sanitize Crew Evaluation, Analysis Crew Evaluation, and RAG Evaluation. Each module focuses on specific aspects such as security validation, project analysis quality, and retrieval-augmented generation performance testing.\n\n",
        # questions[3]: "384",
    }

    # 6) Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 7) Scegli le metriche
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=get_embeddings(),  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = rag_answer(q, chain)
        print(ans)
        print()

if __name__ == "__main__":
    main()