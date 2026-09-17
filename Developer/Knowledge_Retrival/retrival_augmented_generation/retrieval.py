import numpy as np
from Stores.storage import load_question_vectors, load_vectors, save_retrieval_similartiy_search

def cosine_similarties(vec1, vec2):
    vec1= np.array(vec1)
    vec2= np.array(vec2)
    dot= np.dot(vec1, vec2.T)
    norm1= np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2= np.linalg.norm(vec2, axis=1, keepdims=True)
    retrieval= dot/(norm1 @ norm2.T)
    return retrieval

if __name__=="__main__":
    vec1= load_vectors()
    vec2= load_question_vectors()
    result=cosine_similarties(vec1, vec2)
    save_retrieval_similartiy_search(result)
    print("Successfully save the retrival result..")
