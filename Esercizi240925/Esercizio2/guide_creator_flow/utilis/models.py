"""Model and client factory utilities.

This module provides small helper functions to instantiate the project's
LLM and embedding clients (Azure OpenAI wrappers) and a Qdrant client. The
functions read configuration from environment variables, so ensure the
appropriate keys are present before calling them.

Functions
---------
get_embeddings_custom()
    Return an Azure embeddings client configured from environment variables.

get_llm_custom()
    Return an AzureChatOpenAI LLM instance configured from environment.

get_qdrant_client(url)
    Return a ``QdrantClient`` connected to the given url.
"""

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()


def get_embeddings_custom() -> AzureOpenAIEmbeddings:
    """Create an AzureOpenAIEmbeddings instance using environment settings.

    Returns
    -------
    AzureOpenAIEmbeddings
        Embeddings client configured for the workspace.
    """
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )


def get_llm_custom():
    """Create and return an AzureChatOpenAI LLM instance.

    Returns
    -------
    AzureChatOpenAI
        An LLM object configured with deployment name and Azure credentials.
    """
    return AzureChatOpenAI(
        deployment_name="gpt-4o",
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
    )


def get_qdrant_client(url) -> QdrantClient:
    """Return a QdrantClient connected to the specified URL.

    Parameters
    ----------
    url : str
        The endpoint for the Qdrant service (for example "http://localhost:6333").

    Returns
    -------
    QdrantClient
        Client connected to the provided url.
    """
    return QdrantClient(url=url)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

def get_embeddings_custom() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )


def get_llm_custom():
    return AzureChatOpenAI(
        deployment_name="gpt-4o",
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

def get_qdrant_client(url) -> QdrantClient:
    return QdrantClient(url=url)