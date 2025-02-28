from typing import Optional
from pydantic import BaseModel, Field

from constant import models, model_kwargs
from file_loader import Loader, get_num_cpu
from vectordb import VectorDatabase
from llm import get_local_model, get_api_model
from rag import RAG
from utils import select_local_model, select_running_type, get_file_paths

import time

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")


class AnswerQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


def setup_vectordb() -> tuple[VectorDatabase, any]:
    """Set up vector database and retriever"""
    loader = Loader()
    file_paths = get_file_paths()
    reload_pdf = input("Load new PDFs? [Y]/[N] (If this is the first time, please select [Y]): ").lower()
    
    if reload_pdf == "y":
        documents = loader.load(file_paths, workers=get_num_cpu())
        vectordb = VectorDatabase(documents=documents, load_new_vectordb=True)
    else:
        vectordb = VectorDatabase()
    
    return vectordb, vectordb.get_retriever()


def setup_llm():
    """Set up language model based on user choice"""
    running_type = select_running_type()
    
    if running_type == 1:
        model_name = select_local_model()
        return get_local_model(model_name=model_name, **model_kwargs)
    else:
        return get_api_model(**model_kwargs)


def process_query(query: str, rag_chain: any) -> tuple[str, float]:
    """Process a user query and return the answer and processing time"""
    start_time = time.time()
    result = rag_chain(query=query)
    time_taken = time.time() - start_time
    return result, time_taken


def qa_pipeline(rag_chain: any):
    """Main QA interaction loop"""
    while True:
        print("\n" + "="*50)
        question = input("User input (or 'quit' to exit): ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        
        print("\nSearching for relevant information...\n")
        result, time_taken = process_query(question, rag_chain)
        
        print(f"Answer: {result}")
        print(f"Time: {time_taken:.2f} seconds\n")


def main():
    """Main application entry point"""
    # Set up resources
    vectordb, retriever = setup_vectordb()
    llm = setup_llm()
    rag_chain = RAG(llm=llm).get_chain(retriever=retriever)
    
    # Run the QA pipeline
    qa_pipeline(rag_chain)


if __name__ == "__main__":
    main()