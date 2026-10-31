from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from Developer.data.config import DOCUMENT_PATH
from Stores.storage import save_chunks

def load_data(files):
    docs= files.load()
    print(f"Total page is : {len(docs)}")
    return docs

#CharacterTextSplitter, RecursiveCharacterTextSplitter, and TokenTextSplitter

def docsSplitter(docs):
    textSplitter= RecursiveCharacterTextSplitter(
    chunk_size= 850, 
    chunk_overlap= 200,
    length_function= len,
    is_separator_regex= False
    )
    chunks= textSplitter.split_documents(docs)
    print(f"Length of Chunk on the basis of {textSplitter}: ",len(chunks))
    return chunks

if __name__== "__main__":
    raw_data= (DOCUMENT_PATH)
    docs=load_data(raw_data)
    chunks=docsSplitter(docs)
    save_chunks(chunks)
    print("Document Loaded and chunking successfully completed.")

