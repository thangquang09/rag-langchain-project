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
threshold=0.2
K=4