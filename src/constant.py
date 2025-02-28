from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ----- download.py & file_loader.py ------

data_folder = "./data/" # Directory which stores PDF files
chunk_size = 10000 # Can be lower if run local
chunk_overlap = 500

# ----- llm.py -----

models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

model_name = models[0]
max_new_tokens = 1024 # Max token generated
model_kwargs = {
    "temparature": 0
}


# ----- vectordb.py -----

persist_directory = "./chromadb"
load_new_vectordb = False # True: Reinitiate vectordb automatically when run application
threshold=0.5
K=4

# ----- rag.py -----

system = "You are an expert at AI.. Your name is AI Assistant."
human = """
I need your help with a question. Please use only the provided context to answer accurately.

Context:
{context}

Question: {question}

If the context doesn't contain enough information to answer the question completely, please say "I don't have enough information to answer this question" instead of making up information.

Please provide a clear and concise answer based solely on the context provided.
"""

prompt = ChatPromptTemplate([
    ("system", system),
    MessagesPlaceholder("chat_history"),
    ("human", human),
])