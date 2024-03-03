import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".pdf": (PyMuPDFLoader, {}),
}


skipped_files = []
def load_single_document(file_path: str):
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        try:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()
        except AttributeError as e:
            if "'NoneType' object has no attribute 'findall'" in str(e):
                print(f"Skipping file {file_path} due to error: {e}")
                skipped_files.append(file_path)
                return None
            else:
                raise e  # If the error is different, raise it

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files):
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    print(filtered_files)

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                if docs:
                    results.extend(docs)
                pbar.update()

    return results


def ingest_data():
    Data_path = "../../data"
    DB_faiss_path = "../../db_faiss"
    source_directory = Data_path  # Change this to the path of your folder
    documents = load_documents(source_directory, ignored_files=[])

    print(f"Loaded {len(documents)} new documents from {source_directory}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {500} tokens each)")

    # Use HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

    # create and save the local database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_faiss_path)

    if skipped_files:
        print("Skipped the following files due to errors:")
        for path in skipped_files:
            print(path)

if __name__ == '__main__':
    ingest_data()