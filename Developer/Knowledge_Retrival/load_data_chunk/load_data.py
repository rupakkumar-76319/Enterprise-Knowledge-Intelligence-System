from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Developer.data.config import PDF_PATH
from Stores.storage import save_chunks

def load_data(files):
    docs= files.load()
    print(f"Total page is : {len(docs)}")
    return docs

def docsSplitter(docs):
    textSplitter= RecursiveCharacterTextSplitter(
    chunk_size= 1000, 
    chunk_overlap= 200,
    length_function= len,
    is_separator_regex= False
    )
    chunks= textSplitter.split_documents(docs)
    print(len(chunks))
    print(chunks[637])
    return chunks

if __name__== "__main__":
    raw_data= PyPDFLoader(PDF_PATH)
    docs=load_data(raw_data)
    chunks=docsSplitter(docs)
    save_chunks(chunks)
    print("Document Loaded and chunking successfully completed.")

