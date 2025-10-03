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