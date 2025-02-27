# ----- download.py & file_loader.py ------

data_folder = "./data/" # Directory which stores PDF files

# ----- llm.py -----

model_name = "codellama/CodeLlama-7b-Instruct-hf"
max_new_tokens = 256 # Max token generated

# ----- vectordb.py -----

load_new_vectordb = False # True: Reinitiate vectordb automatically when run application
threshold=0.2