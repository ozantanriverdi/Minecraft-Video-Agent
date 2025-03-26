import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "max_tokens": 10
}

response = requests.post(url, headers=headers, json=data)

# Print rate limit headers
print("Request limit:", response.headers.get("x-ratelimit-limit-requests"))
print("Request remaining:", response.headers.get("x-ratelimit-remaining-requests"))
print("Request reset in:", response.headers.get("x-ratelimit-reset-requests"))

print("Token limit:", response.headers.get("x-ratelimit-limit-tokens"))
print("Tokens remaining:", response.headers.get("x-ratelimit-remaining-tokens"))
print("Token reset in:", response.headers.get("x-ratelimit-reset-tokens"))
