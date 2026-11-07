from langchain_classic.embeddings import HuggingFaceEmbeddings
from Users.user_Question import questionAsking
from Stores.storage import save_question_vectors

model_name= "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}  
hf = HuggingFaceEmbeddings(  
    model_name=model_name,  
    model_kwargs=model_kwargs,  
    encode_kwargs=encode_kwargs  
    )
def embed_question(question:str):
    return hf.embed_query(question)
    