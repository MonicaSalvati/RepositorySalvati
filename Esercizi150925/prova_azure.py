import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
api_version = os.getenv("GPT_API_VERSION")
azure_endpoint = os.getenv("GPT_ENDPOINT")
model_name = os.getenv("GPT_NAME")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=model_name
)

print(response.choices[0].message.content)