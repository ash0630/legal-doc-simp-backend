import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)

try:
    models = list(client.models.list())
    for m in models:
        print(m.name)
except Exception as e:
    import traceback
    traceback.print_exc()
