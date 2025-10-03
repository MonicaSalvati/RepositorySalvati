"""Utilities to build datasets and run RAGAS evaluation over RAG outputs.

This module provides helpers to synthesize ground-truth answers with an LLM,
assemble RAGAS-compatible datasets, and execute a suite of RAGAS metrics
using the project's configured LLM and embeddings.

The functions are convenience helpers for offline evaluation and are not
required at runtime for the main Flow. They expect the environment to be
configured with credentials for the project's LLM/embeddings.
"""

import os
from typing import List, Any

import pandas as pd

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
    AnswerRelevancy,
)

from utilis.models import get_embeddings_custom, get_llm_custom
import csv


def get_ground_truth_for_query(query: str, documents: str, topic: str) -> str:
    """Synthesize a reference answer using the LLM and retrieved contexts.

    Parameters
    ----------
    query : str
        The user question for which a reference answer will be generated.
    documents : List[Any]
        Retrieved contexts. Each document is expected to expose the key
        ``"page_content"`` containing text used to guide the generation.
    topic : str
        A short topic label used to structure the synthesized answer.

    Returns
    -------
    str
        A reference answer produced by the LLM, conditioned on the provided
        contexts.
    """
    llm = get_llm_custom()

    # This prompt is intentionally detailed to coax a structured markdown answer
    prompt = (
    f"""    Using the retrieved documents for validation,
            produce a well-structured, maintainable Markdown knowledge article about {topic} answering
            the user question ({query}). Goals:
                1. Title (H1) concise and user-oriented.
                2. Optional Table of Contents if more than 4 H2 sections.
                3. Clear hierarchy: H1 (title), H2 (main sections), H3 (subsections).
                4. Include sections when applicable: Overview, Core Concepts, Step-by-Step / How It Works,
                    Examples, Best Practices, Common Pitfalls / Anti-Patterns, FAQ, Troubleshooting, References.
                5. Integrate only evidence-supported claims; no external unverifiable additions.
                6. Use consistent terminology; avoid redundancy.
                7. Use fenced code blocks only if examples are necessary (specify language).
                8. References section maps includes the source of the documents used to respond, whether is a file name or a url. 
                9. Use only the "source" key of the documents.
                10. If gaps exist, include a "Knowledge Gaps / Further Research" section.
            Style:
                - Neutral, technical, concise.
                - Bullets over long paragraphs where suitable.
                - No promotional language.
                - The content of the file must be in the same language of the input query.
            The expected output is a single valid Markdown document (no enclosing ``` fences) with:

                # <Title>

                (Optional) Table of Contents

                ## Overview
                ...
                
                ## Days (if a travel-guide is asked)
                ...

                ## Restaurants (if a list of Restaurants is asked)
                ...

                ## Hotels (if a list of Hotels is asked)

                ## Examples
                ...

                ## Best Practices
                ...

                ## Common Pitfalls
                ...

                ## FAQ
                - Q: <question>? A: <answer>

                ## References
                - <INSERT HERE THE SOURCE >: <brief relevance>
            
                ## Knowledge Gaps / Further Research (this section only if applicable)
                - <gap>
            Documents: {documents}
        """
    )

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def build_ragas_dataset(
    query: str, documents: str, answer: str, topic: str
) -> List[dict]:
    """Build a list of rows compatible with RAGAS ``EvaluationDataset``.

    Parameters
    ----------
    query : str
        The user input or question being evaluated.
    documents : List[Any]
        Retrieved contexts. Each must include a ``"page_content"`` key.
    answer : str
        The model-generated answer to evaluate against the metrics.
    topic : str
        Topic label used to synthesize reference answers.

    Returns
    -------
    List[dict]
        A list of dataset rows, suitable for creating an
        ``ragas.EvaluationDataset``.
    """
    row = {
        "user_input": query + " Voglio la risposta scritta in un file markdown",
        "retrieved_contexts": documents.split("\n\n"),
        "response": answer,
        "reference": get_ground_truth_for_query(query, documents, topic),
    }
    return [row]


def execute_ragas(user_query: str, documents: str, file_md: str, topic: str) -> None:
    print("Eseguo RAGAS...")
    dataset = build_ragas_dataset(query=user_query, documents=documents, answer=file_md, topic=topic)

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
        AnswerRelevancy(strictness=1),
    ]

    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=get_llm_custom(),
        embeddings=get_embeddings_custom(),
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    out_path = "ragas_results.csv"
    ds_old = pd.read_csv(out_path, sep=";", quoting=csv.QUOTE_ALL) if os.path.exists(out_path) else pd.DataFrame()
    df = pd.concat([ds_old, df], ignore_index=True) if not ds_old.empty else df
    df.to_csv(out_path, index=False, sep=";", quoting=csv.QUOTE_ALL)
    print(f"Salvato: {out_path}")
