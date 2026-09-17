import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from Stores.storage import load_chunks, save_vectors

chunks=load_chunks()


model_name = "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}  
hf = HuggingFaceEmbeddings(  
model_name=model_name,  
model_kwargs=model_kwargs,  
encode_kwargs=encode_kwargs  
)

chunk_text= [c.page_content for c in chunks]
chunks_vectorization= hf.embed_documents(chunk_text)

save_vectors(chunks_vectorization)
