import os
from dotenv import load_dotenv
from google import genai
from Users.user_Question import questionAsking
from Users.user_Question_embedding import embed_question
from Developer.Knowledge_Retrival.retrival_augmented_generation.top_k_retrieval import get_top_k_context
from Developer.Knowledge_Retrival.retrival_augmented_generation.retrieval import cosine_similarties
from Stores.storage import load_vectors, load_chunks
import numpy as np




load_dotenv()

GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API Key is not set in the .env file..")

client= genai.Client()

def create_rag_prompt(context: str, query:str):
    """Combines retrieved context with the user query into a grounded prompt."""
    return f"""You are a helpful enterprise assistant. Answer the question truthfully using only the provided context below. 
    If the answer is not present in the context, reply with: "I do not have sufficient information in the knowledge base to answer this."

    Context: {context}
    Question: {query}
    Answer:"""
    
def generate_answer(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "system_instruction": "You are a factual, concise enterprise knowledge assistant. Answer the question truthfully using only the provided context. If the answer cannot be found in the context, reply with: 'I do not have sufficient information in the knowledge base to answer this.'",
            "temperature": 0.1,
            "max_output_tokens": 2048,
        },
    )
    return response.text

def main():
    query= questionAsking()
    query_vector= embed_question(query)
    scores= cosine_similarties(load_vectors(), query_vector)[:,0]
    top_score=np.argsort(scores)[::-1][:3]
    chunk=load_chunks()
    context= "\n\n".join(chunk[i].page_content for i in top_score)
    answer= generate_answer(create_rag_prompt(context, query))
    print("Your answer is:", answer)
    
if __name__== "__main__":
    main()