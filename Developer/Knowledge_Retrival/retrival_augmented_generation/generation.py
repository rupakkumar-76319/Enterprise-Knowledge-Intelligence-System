import os
from dotenv import load_dotenv
from google import genai
import anthropic
from Users.user_Question import questionAsking
from Users.user_Question_embedding import embed_question
from Developer.Knowledge_Retrival.retrival_augmented_generation.top_k_retrieval import get_top_k_context
from Developer.Knowledge_Retrival.retrival_augmented_generation.retrieval import cosine_similarties
from Stores.storage import load_vectors, load_chunks
import numpy as np




load_dotenv()

ANTHROPIC_API_KEY= os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Gemini API Key is not set in the .env file..")

#client= genai.Client()
client= anthropic.Anthropic()

def create_rag_prompt(context: str, query:str):
    """Combines retrieved context with the user query into a grounded prompt."""
    return f"""You are a helpful enterprise assistant. Answer the question truthfully using only the provided context below. 
    If the answer is not present in the context, reply with: "I do not have sufficient information in the knowledge base to answer this."

    Context: {context}
    Question: {query}
    Answer:"""
    
# def generate_answer(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
#     response = client.models.generate_content(
#         model=model_name,
#         contents=prompt,
#         config={
#             "system_instruction": "You are a factual, concise enterprise knowledge assistant. Answer the question truthfully using only the provided context. If the answer cannot be found in the context, reply with: 'I do not have sufficient information in the knowledge base to answer this.'",
#             "temperature": 0.1,
#             "max_output_tokens": 2048,
#         },
#     )
#     return response.text

def generate_answer(prompt: str, model_name: str = "claude-opus-5-5") -> str:
    response = client.beta.messages.create(
        model=model_name,
        max_tokens=16000,
        system="You are a factual, concise enterprise knowledge assistant. Answer the question truthfully using only the provided context. If the answer cannot be found in the context, reply with: 'I do not have sufficient information in the knowledge base to answer this.'",
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "Claude declined to answer this question."
    return "".join(b.text for b in response.content if b.type == "text")


def main():
    query= questionAsking()
    query_vector= embed_question(query)
    scores= cosine_similarties(load_vectors(), query_vector)[:,0]
    top_score=np.argsort(scores)[::-1][:50]
    chunk=load_chunks()
    context= "\n\n".join(chunk[i].page_content for i in top_score)
    answer= generate_answer(create_rag_prompt(context, query))
    print("Your answer is:", answer)
    
if __name__== "__main__":
    main()