from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ----- download.py & file_loader.py ------

data_folder = "./data/" # Directory which stores PDF files
chunk_size = 300 # Can be lower if run local
chunk_overlap = 0

# ----- llm.py -----

models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

model_name = models[0]
max_new_tokens = 1024 # Max token generated
model_kwargs = {
    "temparature": 0.7
}


# ----- vectordb.py -----

persist_directory = "./chromadb"
load_new_vectordb = False # True: Reinitiate vectordb automatically when run application
K=4

# ----- rag.py -----

system = "You are an expert at AI.. Your name is AI Assistant."
human = """
I need your help with a question.

Context:
{context}

Question: {question}

If the provided context doesn't contain sufficient information to fully answer the question:
1. First acknowledge what information is missing from the context
2. Then try to answer based on our conversation history if relevant
3. If neither the context nor our chat history helps, please state "I don't have enough information to answer this question"

Please provide a clear and concise answer, prioritizing information from the provided context when available.
"""

prompt = ChatPromptTemplate([
    ("system", system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", human),
])