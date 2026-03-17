import requests
import urllib.request
import os

pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
pdf_path = "sample_test.pdf"

print("Downloading dummy PDF...")
urllib.request.urlretrieve(pdf_url, pdf_path)

print("Testing /upload endpoint...")
with open(pdf_path, "rb") as f:
    response = requests.post("http://localhost:8000/upload", files={"file": f})
    
print("Upload response:", response.status_code, response.text)

print("\nTesting /ask endpoint...")
response2 = requests.post("http://localhost:8000/ask", json={"question": "What does this document say?"})
try:
    print("Ask response:", response2.status_code, response2.text)
except UnicodeEncodeError:
    print("Ask response (Encoded):", response2.status_code, response2.text.encode('utf-8'))
