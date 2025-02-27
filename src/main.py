from constant import models
from file_loader import Loader, file_paths, get_num_cpu
from vectordb import VectorDatabase
from llm import get_local_model, get_api_model
from rag import RAG
from utils import select_local_model, select_running_type

from pydantic import BaseModel, Field
import time


class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class AnswerQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


# ----- Load Resources -----

## ----- PDF REAEDER ------
loader = Loader()
reload_pdf = input("Load new PDFs? [Y]/[N] (If this is the first time, please select [Y]): ")

if reload_pdf.lower() == "y":
    documents = loader.load(file_paths, workers=get_num_cpu())
    load_new_vectordb = True
    vectordb = VectorDatabase(documents=documents, load_new_vectordb=load_new_vectordb)
else:
    documents = None
    vectordb = VectorDatabase()

retriever = vectordb.get_retriever()

## ----- Model Selection ------

running_type = select_running_type()

if running_type == 1:
    model_name = select_local_model()
    llm = get_local_model(model_name=model_name, temparature=0.7)
elif model_name == 2:
    llm = get_api_model()


## ----- Make RAG Chain -----
rag_chain = RAG(llm=llm).get_chain(retriever=retriever)

# ------ Pipeline ------
def QAPipeline():
    while True:
        print("\n" + "="*50)
        question = input("Ask me anything (or 'quit' to exit): ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        user_question = InputQA(question=question)
        print("\nSearching for relevant information...")
        
        start_time = time.time()
        result = rag_chain(query=user_question.question)
        time_taken = time.time() - start_time
        
        answer = AnswerQA(answer=result)
        
        print("\nResults:")
        print("-"*50)
        print(f"Q: {user_question.question}")
        print(f"A: {answer.answer}")
        print(f"Time: {time_taken:.2f} seconds")

if __name__ == "__main__":
    QAPipeline()