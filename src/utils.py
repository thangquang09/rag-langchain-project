from constant import models, data_folder, persist_directory, huggingface_key
from download import download_pdfs
from file_loader import Loader, get_num_cpu, get_file_paths
from vectordb import VectorDatabase

from dotenv import load_dotenv
import os
import requests
def check_data_exists():
    """Check if data folder and persist directory exist.
    Returns:
        int: 1 if data_folder doesn't exist but persist_directory does
             2 if persist_directory doesn't exist but data_folder does
             3 if neither data_folder nor persist_directory exist
             0 if both exist
    """
    data_missing = not os.path.exists(data_folder)
    vector_db_missing = not os.path.exists(persist_directory)
    
    if data_missing and vector_db_missing:
        return 3
    elif data_missing:
        return 1
    elif vector_db_missing:
        return 2
    return 0


def initial_data(option: int):
    """Initialize data based on the option.
    Args:
        option: 1 = download PDFs, 2 = create vector DB, 3 = both
    """
    if option in (1, 3):
        download_pdfs()
    
    if option in (2, 3):
        file_paths = get_file_paths()
        loader = Loader()
        documents = loader.load(file_paths, workers=get_num_cpu())
        VectorDatabase(documents=documents, load_new_vectordb=True)


def check_model_exists(model_name):

    if huggingface_key is None:
        print("Please create your HuggingFace Read token!")
        user_input = input("Do you want to cancel model checker? [Y]/[N]: ")
        if user_input.lower() == "y":
            # cancel
            return True
        else:
            exit(0)


    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {"Authorization": f"Bearer {huggingface_key}"}
    response = requests.get(url, headers=headers)

    
    if response.status_code == 200:
        print(f"Model '{model_name}' exists on Hugging Face.")
        return True
    elif response.status_code == 404:
        print(f"Model '{model_name}' not exists on Hugging Face.")
        return False
    else:
        print(f"Error while checking: {response.status_code}")
        return False


def select_local_model() -> str:
    print("Which model do you want to use?")

    for idx, model in enumerate(models):
        print(f"{[idx]}: {model}")
    choice = input("Input a model id (you can paste any model name on HuggingFace instead): ")
    try:
        choice_idx = int(choice)
        if 0 <= choice_idx < len(models):
            model_name = models[choice_idx]
            return model_name # dont need to check
        else:
            print(f"Index {choice_idx} is out of range. Using the default model.")
            model_name = models[0]
    except ValueError:
        # If not a number, assume it's a model name
        model_name = choice
        print(f"Using custom model: {model_name}")

    if check_model_exists(model_name):
        print(f"Selected model: {model_name}")
        return model_name
    else:
        return select_local_model()

def select_running_type() -> int:
    while True:
        running_type = input("Do you want to run local or call API?\n[1]: local\n[2]: API\nChoice: ")
        try:
            running_type = int(running_type)
            if running_type == 1 or running_type == 2:
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid choice")
    return running_type
