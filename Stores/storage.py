import json
from langchain_core.documents import Document

def save_chunks(chunks, path="Developer/data/chunks.json"):
    data= [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(path, "w", encoding= "utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def load_chunks(path='Developer/data/chunks.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data=json.load(f)
    return [Document (page_content=d['content'], metadata=d['metadata']) for d in data]

def save_vectors(vectors, path="Developer/data/vectors.json"):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vectors, f)
        
def load_vectors(path='Developer/data/vectors.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_question_vectors(question_vectors, path='Developer/data/question_vectors.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(question_vectors, f)
        
def load_question_vectors(path='Developer/data/question_vectors.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_retrieval_similartiy_search(retieval, path='Developer/data/retrieval.json'):
    # Convert numpy to python list so that it stores in json format..
    if hasattr(retieval, 'tolist'):
        retieval= retieval.tolist()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(retieval, f)
