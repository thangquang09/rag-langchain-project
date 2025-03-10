from constant import data_folder, chunk_size, chunk_overlap
from download import download_pdfs

from typing import List, Union
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import os
import multiprocessing



# ------------------ Useful Function ----------------------------
def get_file_paths():
    return [os.path.join(data_folder, path) for path in os.listdir(data_folder)]

def remove_non_utf8_character(page_content: str) -> str:
    processed_content = ''.join(char for char in page_content if ord(char) < 128)

    return processed_content

def load_pdf_file(file_path: str) -> list:
    pages = PyPDFLoader(file_path, extract_images=True).load()
    for page in pages:
        page.page_content = remove_non_utf8_character(page.page_content)
    return pages

def get_num_cpu():
    return multiprocessing.cpu_count()

def is_valid_pdf(file_path: str) -> bool:
    if not file_path.lower().endswith('.pdf'):
        return False
    if not os.path.exists(file_path):
        return False
    return True

# ------------------- Class --------------------
class BaseLoader:
    def __init__(self, ) -> None:
        self.num_cpu = get_num_cpu()
    
    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    def __init__(self, ) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs) -> Document:
        num_cpu = kwargs["workers"]
        with multiprocessing.Pool(processes=num_cpu) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="files") as pbar:
                for result in pool.imap_unordered(load_pdf_file, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ['\n\n', '\n', '. ', '! ', '? ', ':', ';', ' '],
        chunk_size: int = chunk_size,
        chunk_overlap: int = chunk_overlap
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    def __call__(self, documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(
        self,
        file_type: str = "pdf",
        split_kwargs: dict = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        } 
    ) -> None:
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.text_splitter = TextSplitter(**split_kwargs)
    
    def load(
        self,
        pdf_files: Union[str, List[str]],
        workers: int = 1
    ) -> Document:
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        pdf_files = [file_path for file_path in pdf_files if is_valid_pdf(file_path)]
        
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_splitted = self.text_splitter(doc_loaded)

        return doc_splitted

# -------------- Main -----------------------------

def main():
    file_paths = get_file_paths()
    num_cpu = get_num_cpu()
    loader = Loader()

    doc_splitted = loader.load(file_paths, workers=num_cpu)


if __name__ == "__main__":
    main()
