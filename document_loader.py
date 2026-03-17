import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google import genai

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def extract_text_with_gemini(file_path: str, mime_type: str) -> str:
    """Uses Gemini API to extract text from images or scanned PDFs."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set!")
        return ""
    try:
        client = genai.Client()
        
        logger.info(f"Uploading {file_path} to Gemini for OCR...")
        uploaded_file = client.files.upload(file=file_path, config={'mime_type': mime_type})
        
        prompt = "Extract all text exactly as it appears in this document. Do not summarize or add any other commentary. Just return the extracted text."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[uploaded_file, prompt]
        )
        
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception as e:
            logger.warning(f"Could not delete uploaded file from Gemini APIs: {e}")
            
        return response.text
    except Exception as e:
        logger.error(f"Failed to extract text with Gemini OCR: {e}")
        return ""

def process_and_chunk_document(file_path: str, filename: str) -> list:
    """
    Extracts text (using OCR if necessary) from a saved file, and splits it into chunks.
    """
    documents = []
    ext = os.path.splitext(filename)[1].lower()
    needs_ocr = False
    mime_type = "application/pdf"
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Heuristic: if very little text is extracted compared to the number of pages, it's likely a scanned PDF
        total_text_length = sum(len(doc.page_content.strip()) for doc in documents)
        if len(documents) > 0 and total_text_length < 50 * len(documents):
            logger.info("PDF appears to be scanned or image-based. Falling back to Gemini OCR.")
            needs_ocr = True
    elif ext in [".png", ".jpg", ".jpeg"]:
        needs_ocr = True
        mime_type = f"image/{ext[1:].replace('jpg', 'jpeg')}"
        
    if needs_ocr:
        extracted_text = extract_text_with_gemini(file_path, mime_type)
        if extracted_text and extracted_text.strip():
            documents = [Document(page_content=extracted_text, metadata={"source": file_path})]
        elif ext != ".pdf":
            documents = [] # Failed to extract from image
            
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
