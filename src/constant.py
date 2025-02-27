# ----- download.py & file_loader.py ------

data_folder = "./data/" # Directory which stores PDF files

# ----- llm.py -----

models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

model_name = models[0]
max_new_tokens = 1024 # Max token generated

# ----- vectordb.py -----

load_new_vectordb = False # True: Reinitiate vectordb automatically when run application
threshold=0.2