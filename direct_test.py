import os
import requests
import urllib.request
import traceback

from document_loader import save_and_chunk_pdf
from vector_store import store_chunks_in_vectorstore, retrieve_top_k
from rag_pipeline import generate_response

def run():
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    pdf_path = "sample_test.pdf"

    print("Downloading dummy PDF...")
    urllib.request.urlretrieve(pdf_url, pdf_path)

    try:
        with open(pdf_path, "rb") as f:
            contents = f.read()
        print(f"File Size: {len(contents)} bytes")
        
        chunks = save_and_chunk_pdf(contents, "sample_test.pdf")
        print(f"Extracted {len(chunks)} chunks.")
        
        store_chunks_in_vectorstore(chunks)
        print("Stored chunks in Vectorstore.")
        
        results = retrieve_top_k("What does this document say?")
        print(f"Retrieved {len(results)} chunks.")
        
        if results:
            ans = generate_response("What does this document say?", results, [])
            print(f"Answer: {ans}")
    except Exception as e:
        print("Error encountered:")
        traceback.print_exc()

if __name__ == "__main__":
    run()
