from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
azure_endpoint = os.getenv("ENDPOINT")
model_name = os.getenv("MODEL_NAME")

client = AzureOpenAI(
    api_key = api_key,
    api_version = api_version,
    azure_endpoint = azure_endpoint
)

text = "Today is a good day"

response = client.embeddings.create(
    model=model_name,
    input=text
)

embedding = response.data[0].embedding

print("Embedding generato:")
print(embedding[:10], "...")
