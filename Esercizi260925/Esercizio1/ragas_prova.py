"""
RAGAS evaluation utilities.

This module provides functions to build RAGAS-compatible datasets and execute
evaluation metrics for Retrieval-Augmented Generation (RAG) workflows.
It leverages the `ragas` library to compute context precision, recall,
faithfulness, answer relevancy, and correctness.
"""

from typing import List, Any
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" on retrieved chunks
    context_recall,      # coverage of relevant chunks
    faithfulness,        # grounding of the answer in the context
    answer_relevancy,    # relevance of answer vs question
    answer_correctness,  # use only if ground_truth is available
)
from utilis.models import get_embeddings, get_llm


def get_ground_truth_for_query(query: str, documents: List[Any]) -> str:
    """
    Generate a ground-truth answer for a given query and context documents.

    This function uses the configured LLM to produce a reference answer based
    on the retrieved documents, which can be used in RAGAS evaluation metrics
    like `answer_correctness`.

    Parameters
    ----------
    query : str
        The user query or question to answer.
    documents : List[Any]
        List of context documents, each containing a "page_content" field.

    Returns
    -------
    str
        The generated ground-truth response for the given query.
    """
    llm = get_llm()
    joined = "\n---\n".join([document["page_content"] for document in documents])

    prompt = f"""Sei un assistente che crea file.md per rispondere alla domanda dell'utente.
Domanda: {query}

Contesti:
{joined}

Risposta:"""

    response = llm.invoke(prompt)
    return response.content


def build_ragas_dataset(query: str, documents: List[Any], answer: str) -> List[dict]:
    """
    Build a RAGAS-compatible dataset for evaluation.

    Each row contains the query, retrieved contexts, model response, and
    optional reference answer (ground truth).

    Parameters
    ----------
    query : str
        The user query or question.
    documents : List[Any]
        List of context documents with a "page_content" field.
    answer : str
        The model-generated response to the query.

    Returns
    -------
    List[dict]
        A dataset compatible with RAGAS evaluation functions, where each dict
        contains the keys: "user_input", "retrieved_contexts", "response", "reference".
    """
    row = {
        "user_input": query,
        "retrieved_contexts": [document["page_content"] for document in documents],
        "response": answer,
        "reference": get_ground_truth_for_query(query, documents)
    }

    return [row]


def execute_ragas(user_query: str, documents: List[Any], file_md: str) -> None:
    """
    Execute RAGAS evaluation for a single query and its generated response.

    This function builds the dataset, computes metrics using `ragas.evaluate`,
    prints a summary to the console, and saves detailed results to a CSV file.

    Parameters
    ----------
    user_query : str
        The user question or query to evaluate.
    documents : List[Any]
        List of context documents for the query.
    file_md : str
        The model-generated answer to the query.

    Returns
    -------
    None
        Prints metric results and saves them to 'ragas_results.csv'.
    """
    print("Eseguo RAGAS...")
    dataset = build_ragas_dataset(
        query=user_query,
        documents=documents,
        answer=file_md
    )
    print("Dataset per RAGAS:", dataset)

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    print("Eseguo valutazione RAGAS...")
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness
    ]

    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=get_llm(),                  # LangChain LLM instance
        embeddings=get_embeddings(),    # Embedding model instance
    )

    df = ragas_result.to_pandas()
    cols = [
        "user_input",
        "response",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy"
    ]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # Optionally save results for human review
    df.to_csv("ragas_results.csv", index=False, sep=";")
    print("Salvato: ragas_results.csv")
