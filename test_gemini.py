import os
from google import genai
from google.genai import types
import traceback

api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBVPO5W56vJNVNNQrrN6qylHx57oStqJGs")

client = genai.Client(api_key=api_key)

contents = [
    {"role": "user", "parts": [{"text": "Hello"}]}
]

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction="You are helpful."
        )
    )
    print("SUCCESS")
    print(response.text)
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(str(e))
    traceback.print_exc()
