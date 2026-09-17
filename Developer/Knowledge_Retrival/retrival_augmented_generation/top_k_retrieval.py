import numpy as np
import json


def get_top_k_context(similarity_matrix_path, raw_chunk_path, top_k=3):
    
    with open(similarity_matrix_path, 'r', encoding='utf-8') as f:
        scores= np.array(json.load(f))
        
    mean_scores= np.mean(scores, axis=1)
    top_indices= np.argsort(mean_scores)[::-1][:top_k]
    
    with open(raw_chunk_path, 'r', encoding='utf-8') as f:
        raw_chunk= json.load(f)
        
    
    context_list=[]
    for idx in top_indices:
        item=(
            raw_chunk[idx]
            if isinstance (raw_chunk, list)
            else raw_chunk.get(str(idx), raw_chunk.get(int(idx)))
        )
        if isinstance(item, dict):
            item= item.get("text", str(item))
        context_list.append(str(item))
    
    augmented_context= "\n\n".join(context_list)
    return augmented_context

if __name__=="__main__":
    similarity_matrix_path=r'Developer\data\retrieval.json'
    raw_chunk_path=r'Developer\data\chunks.json'
    extracted_context=get_top_k_context(similarity_matrix_path, raw_chunk_path, top_k=3)
    print("Extracted Context data:\n",extracted_context )
        
        