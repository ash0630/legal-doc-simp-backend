import os
import time
import re
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Legal AI Assistant.

You help users understand legal documents and answer legal questions.

Rules:
- If greeting → respond normally
- If general question → explain simply
- If document question → use context

STRICT:
- Never dump raw text
- Always summarize
- Use structured format
"""

FORMATTING_INSTRUCTIONS = """
Your response MUST use the following exact structure and markdown headings (omit sections not applicable, but maintain this structure):

### ✅ Summary
(1-2 lines summarizing the answer)

### 📌 Key Points
- (Bullet point 1)
- (Bullet point 2)

### ⚠️ Risks (if applicable)
- (Highlight any risks or state "No major risks identified")

### 🧠 Simple Explanation
(Plain English explanation)
"""

def detect_intent(query: str) -> str:
    query_lower = query.lower().strip()
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "sup", "howdy"}
    
    if query_lower in greetings or any(query_lower.startswith(g + " ") for g in greetings):
        return "greeting"
        
    doc_keywords = ["document", "doc", "contract", "agreement", "uploaded", "file", "text", "page", "section", "clause", "pdf"]
    if any(keyword in query_lower.split() for keyword in doc_keywords) or len(query_lower.split()) > 8:
        return "rag"
        
    return "general"

def clean_context(chunks: list, top_k: int = 5) -> str:
    if not chunks:
        return ""
    top_chunks = chunks[:top_k]
    cleaned_texts = []
    for chunk in top_chunks:
        text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_texts.append(text)
    return "\n\n".join(cleaned_texts)

def generate_response(query: str, chunks: list, chat_history: list = None) -> str:
    if chat_history is None:
        chat_history = []
        
    intent = detect_intent(query)
    logger.info(f"Detected intent: {intent}")
    
    if intent == "greeting":
        return "### ✅ Summary\nHello! I am your Legal AI Assistant.\n\n### 📌 Key Points\n- I can answer general legal questions.\n- I can simplify uploaded legal documents.\n\n### 🧠 Simple Explanation\nFeel free to ask me anything about the law or upload a document for us to review together!"
        
    context_text = ""
    if intent == "rag":
        context_text = clean_context(chunks)
        if not chunks:
            return "### ✅ Summary\nNo document found.\n\n### 📌 Key Points\n- You asked a document-related question.\n- No document is currently uploaded or relevant.\n\n### 🧠 Simple Explanation\nI could not find this in the document. Please ensure you have uploaded a document or ask a general legal question instead."
            
    history_str = ""
    for msg in chat_history[-5:]:
        role = msg.get("role", "User").capitalize()
        history_str += f"{role}: {msg.get('content', '')}\n"

    user_input = f"{FORMATTING_INSTRUCTIONS}\n\nContext:\n{context_text}\n\nUser:\n{query}\n\nChat History:\n{history_str}\n\nAnswer:"
    
    try:
        if not hasattr(generate_response, "client"):
            generate_response.client = genai.Client()

        response = generate_response.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_input,
            config=types.GenerateContentConfig(
                temperature=0.3,
                system_instruction=SYSTEM_PROMPT,
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate AI response: {e}")
        return f"### ✅ Summary\nError connecting to AI.\n\n### 📌 Key Points\n- Connection failed.\n- Details: {str(e)}\n\n### 🧠 Simple Explanation\nThere was an issue reaching the AI backend. Please check your API key and connection."

def generate_document_summary(context: str) -> str:
    prompt = f"""You are a Legal Document Simplifier AI assistant.
Please read the provided text from a legal document and provide a DETAILED and COMPREHENSIVE explanation of its contents in simple English.
Focus on identifying:
1. The type of document.
2. The involved parties and their roles.
3. The main purpose and high-level obligations.
4. Any key dates, deadlines, or monetary values mentioned.
Make the explanation thorough enough that a non-lawyer can fully understand the document's substance without reading it.
Use clear headings and bullet points.

Document Text (excerpt):
{context[:10000]}
"""
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate document summary: {e}")
        return "I could not automatically summarize the document at this time. Please proceed to ask specific questions about the clauses."
