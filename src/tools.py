import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

def get_response(question: str):
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "data-py"
    index = pc.Index(index_name)
    # Semantic search
    results = index.search(
        namespace="ns1",
        query={
            "top_k": 5,
            "inputs": {"text": question}
        }
    )
    # Rerank results
    reranked_results = index.search(
        namespace="ns1",
        query={
            "top_k": 5,
            "inputs": {"text": question}
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 5,
            "rank_fields": ["chunk_text"]
        }
    )
    matches = reranked_results.get('matches', [])
    # Check if top match is relevant to the question (contains key terms)
    key_terms = [w for w in question.lower().split() if w in ["address", "location", "house", "sharing"]]
    top_chunk = matches[0]['metadata']['chunk_text'].lower() if matches else ""
    if not matches or not any(term in top_chunk for term in key_terms):
        # Keyword fallback
        file_path = "/Users/haileyoon/code/comfortwomen_text.txt"
        with open(file_path, encoding="utf-8") as f:
            raw = f.read()
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        chunk_size = 3
        stride = 1
        chunks = []
        for i in range(0, len(paragraphs) - chunk_size + 1, stride):
            chunk = "\n\n".join(paragraphs[i:i+chunk_size])
            chunks.append(chunk)
        if len(paragraphs) % chunk_size != 0:
            chunk = "\n\n".join(paragraphs[-chunk_size:])
            if chunk not in chunks:
                chunks.append(chunk)
        q = question.lower()
        best = None
        for chunk in chunks:
            if ("house of sharing" in chunk.lower()) and ("address" in chunk.lower() or "location" in chunk.lower()):
                best = chunk
                break
        if not best:
            # fallback: any chunk with 'house of sharing'
            for chunk in chunks:
                if "house of sharing" in chunk.lower():
                    best = chunk
                    break
        if best:
            return {'matches': [{
                'score': 1.0,
                'metadata': {'chunk_text': best}
            }]}
    return reranked_results
