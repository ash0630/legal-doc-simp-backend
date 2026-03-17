from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import shutil
from dotenv import load_dotenv

load_dotenv()

from document_loader import process_and_chunk_document
from vector_store import store_chunks_in_vectorstore, retrieve_top_k
from rag_pipeline import generate_document_summary, generate_response, detect_intent

import traceback

logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Document Simplifier API")

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173") # Default to local Vite port
# Split by comma if multiple URLs are provided
allowed_origins = [url.strip() for url in FRONTEND_URL.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    chat_history: List[Message] = []

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    allowed_extensions = (".pdf", ".png", ".jpg", ".jpeg")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF and image files (.png, .jpg, .jpeg) are supported.")
    
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        # Stream file to disk to prevent OOM
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Extract and chunk PDF
        chunks = process_and_chunk_document(file_path, file.filename)
        
        # 2. Store in vector database
        store_chunks_in_vectorstore(chunks)
        
        # 3. Generate a high-level document summary
        full_text = " ".join([chunk.page_content for chunk in chunks])
        summary = generate_document_summary(full_text)
        
        return {
            "message": "File uploaded and processed successfully", 
            "chunks_count": len(chunks),
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error processing upload: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the file.")

@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        intent = detect_intent(question)
        
        results = []
        if intent == "rag":
            try:
                results = retrieve_top_k(question)
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}. Falling back to chat mode.")
        
        context_used = [doc.page_content for doc in results[:5]] if results else []
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        simplified_answer = generate_response(question, results, history_dicts)
        
        return {
            "answer": simplified_answer,
            "context_used": context_used
        }
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your question.")

@app.post("/reset")
def reset_backend_state(x_admin_token: Optional[str] = Header(None)):
    admin_secret = os.environ.get("ADMIN_SECRET_TOKEN", "super-secret-default-token")
    if x_admin_token != admin_secret:
        raise HTTPException(status_code=401, detail="Unauthorized client")
        
    try:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
            
        # Also clean up uploads folder
        if os.path.exists("uploads"):
            shutil.rmtree("uploads")
            os.makedirs("uploads")
            
        return {"message": "Reset successful"}
    except Exception as e:
        logger.error(f"Error resetting state: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error during reset.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
