"""
Utilities for initializing Azure OpenAI LLM and embeddings.

This module provides helper functions to create instances of Azure OpenAI
chat models and embeddings using environment variables defined in a `.env` file.
"""

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Load environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")


def get_embeddings() -> AzureOpenAIEmbeddings:
    """
    Initialize an Azure OpenAI embeddings model.

    Returns
    -------
    AzureOpenAIEmbeddings
        An embeddings instance configured with the Azure OpenAI endpoint,
        API key, and API version.

    Notes
    -----
    - Reads configuration from environment variables:
      `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`.
    """
    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
    )


def get_llm() -> AzureChatOpenAI:
    """
    Initialize an Azure OpenAI chat model (LLM).

    Returns
    -------
    AzureChatOpenAI
        An Azure Chat LLM instance configured with deployment, endpoint, key, and API version.

    Notes
    -----
    - Deployment name is currently hardcoded as 'gpt-4o'.
    - Reads endpoint, API key, and API version from environment variables.
    """
    return AzureChatOpenAI(
        deployment_name="gpt-4o",
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
    )
