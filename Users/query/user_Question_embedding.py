from langchain_classic.embeddings import HuggingFaceEmbeddings
from Users.query.user_Question import questionAsking
from Stores.storage import save_question_vectors



model_name= "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}  

if __name__== "__main__":
    hf = HuggingFaceEmbeddings(  
    model_name=model_name,  
    model_kwargs=model_kwargs,  
    encode_kwargs=encode_kwargs  
    )
    question= questionAsking()
    question_vectorization= hf.embed_documents(question)
    save_question_vectors(question_vectorization)

