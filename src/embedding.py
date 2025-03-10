from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

class TaskOptimizedEmbeddings:
    def __init__(self, api_key=None):
        self.api_key = api_key
        
        # Embedding models for different tasks
        self.document_embedder = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            task_type="retrieval_document",
            model="models/embedding-001"
        )
        
        self.query_embedder = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            task_type="retrieval_query",
            model="models/embedding-001"
        )
        
        self.similarity_embedder = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            task_type="semantic_similarity",
            model="models/embedding-001"
        )
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # for task_type = "retrieval_document"
        return self.document_embedder.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.query_embedder.embed_query(text)
    
    def embed_for_similarity(self, texts: List[str]) -> List[List[float]]:
        return self.similarity_embedder.embed_documents(texts)