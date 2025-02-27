from constant import data_folder

import os
import wget
from tqdm import tqdm


def make_data_folder(data_folder:str = data_folder):
    if not os.path.exists(data_folder):
        print("Making Data Folder")
        os.makedirs(data_folder)

def download_pdfs(data_folder:str = data_folder):
    make_data_folder(data_folder)
    print("Downloading Data")
    for file_link in tqdm(file_links):
        file_name = f"{os.path.join(data_folder, file_link['title'])}.pdf"
        if not os.path.exists(file_name):
            wget.download(file_link['url'], out=file_name)
    print("Downloaded Data")

file_links = [
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "url": "https://arxiv.org/pdf/2302.13971"
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401"
    },
    {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "url": "https://arxiv.org/pdf/2305.14314"
    },
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762"
    },
    {
        "title": "Vat ly lop 12",
        "url": "https://84864c160d.vws.vegacdn.vn//Data/hcmedu/thptnguyentatthanh/2021_9/ly-12-co-ban_79202192648.pdf"
    }
]


if __name__ == "__main__":
    download_pdfs()