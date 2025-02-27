from file_loader import Loader, file_paths, get_num_cpu
from vectordb import VectorDatabase
from llm import get_model
from rag import RAG

from pydantic import BaseModel, Field
import time

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class AnswerQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


# ----- Load Resources -----
loader = Loader()
reload_pdf = input("Load new PDFs? [Y]/[N]: ")

if reload_pdf.lower() == "y":
    documents = loader.load(file_paths, workers=get_num_cpu())
    load_new_vectordb = True
    vectordb = VectorDatabase(documents=documents, load_new_vectordb=load_new_vectordb)
else:
    documents = None
    vectordb = VectorDatabase()

retriever = vectordb.get_retriever()
model = get_model(temparature=0.1)
rag_chain = RAG(llm=model).get_chain(retriever=retriever)

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
        result = rag_chain.invoke(user_question.question)
        time_taken = time.time() - start_time
        
        answer = AnswerQA(answer=result)
        
        print("\nResults:")
        print("-"*50)
        print(f"Q: {user_question.question}")
        print(f"A: {answer.answer}")
        print(f"Time: {time_taken:.2f} seconds")

if __name__ == "__main__":
    QAPipeline()