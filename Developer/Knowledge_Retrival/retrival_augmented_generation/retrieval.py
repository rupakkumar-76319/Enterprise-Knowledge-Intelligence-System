import numpy as np

def cosine_similarties(vec1, vec2):
    vec1= np.array(vec1)
    vec2= np.array(vec2)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    dot= np.dot(vec1, vec2.T)
    norm1= np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2= np.linalg.norm(vec2, axis=1, keepdims=True)
    retrieval= dot/(norm1 @ norm2.T)
    return retrieval

