from constant import load_new_vectordb, threshold, K, persist_directory

from typing import Union, List
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import warnings

class VectorDatabase:
    def __init__(
        self,
        documents: List[Document] = None,
        vectordb_class: Union[type(Chroma), type(FAISS)] = Chroma,
        embedding = HuggingFaceEmbeddings(),
        persist_directory: str = persist_directory,
        load_new_vectordb: bool = load_new_vectordb,
        threshold: int = threshold
    ):
    
        self.vectordb_class = vectordb_class
        self.embedding = embedding
        self.persist_directory = persist_directory
        self.threshold = threshold
        
        # Check if vector DB exists and whether to load it
        if os.path.exists(self.persist_directory) and not load_new_vectordb:
            self.vectordb = self.load_vectordb()
        elif documents is not None:
            self.vectordb = self.create_vectordb(documents)
        else:
            raise ValueError("Either documents must be provided or a vector database must exist at the persist_directory and load_new_vectordb must be False")

    def create_vectordb(self, documents: List[Document]):
        """Create vector database from documents"""
        print("Creating Vector Database")
        self.vectordb = self.vectordb_class.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        return self.vectordb

    def load_vectordb(self):
        """Load existing vector database from persist directory"""
        if self.vectordb_class == Chroma:
            return self.vectordb_class(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
        else:  # For FAISS
            return self.vectordb_class.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embedding
            )
            
    def get_retriever(self, search_type="similarity_score_threshold", **kwargs):
        """Get retriever with similarity score threshold"""
        search_kwargs = {
            "k": kwargs.get("k", K),
            "score_threshold": self.threshold,
        }
        
        # Create a retriever that suppresses all warnings
        retriever = self.vectordb.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        # Create a wrapper function to suppress warnings during retrieval
        def silent_retriever(query):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
                try:
                    return retriever.invoke(query)
                except Exception:
                    # Return empty list if anything goes wrong
                    return []
        
        # Return the modified silent retriever
        return silent_retriever

if __name__ == "__main__":
    pass