import os
from langchain_community.vectorstores import FAISS
from embeddings import get_embedding_model

FAISS_INDEX_DIR = "faiss_index"

def store_chunks_in_vectorstore(chunks):
    """
    Stores document chunks into FAISS.
    """
    embeddings_model = get_embedding_model()
    # Create and persist the vector store
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore

def get_vectorstore():
    """
    Loads the persistent FAISS vector store.
    """
    embeddings_model = get_embedding_model()
    # allow_dangerous_deserialization is needed for loading a saved FAISS index in newer versions
    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings_model, allow_dangerous_deserialization=True)
    return vectorstore

def retrieve_top_k(query: str, k: int = 3) -> list:
    """
    Retrieves the top k most relevant chunks for a given query.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results
