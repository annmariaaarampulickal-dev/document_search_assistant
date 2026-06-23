from openai import OpenAI
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# Bypass SSL verification for corporate network
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.choices[0].message.content)