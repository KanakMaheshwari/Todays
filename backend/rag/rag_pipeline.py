import asyncio
import json
import os
from turtledemo.penrose import start

import numpy as np
import time
import faiss
import hashlib
import ollama



from sentence_transformers import SentenceTransformer
from together import Together


dimension = 384

from backend.data.config import (
    VECTOR_STORE_PATH,
    PROCESSED_ARTICLES_PATH,
    CHUNK_INDEX_DIR,
    CACHE_DIR,
)



similarity_search_index = None
def load_stored_faiss_vector():
    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            print(f"FAISS index not found at {VECTOR_STORE_PATH}. Please run article_processor.py first.")
            return
        else:
            return faiss.read_index(VECTOR_STORE_PATH)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception as e:
    print(f"Failed to create or access {CACHE_DIR}: {e}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_processed_article():
    try:
        if os.path.exists(PROCESSED_ARTICLES_PATH):
            with open(PROCESSED_ARTICLES_PATH, "r",encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Failed to load processed articles from {PROCESSED_ARTICLES_PATH}: {e}")
        return []

async def generate_embedding(text):
    try:
        return np.array(embedding_model.encode(text[:512], show_progress_bar=False), dtype="float32")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    
async def filter_by_cluster(query, processed_articles, index):
    start_time = time.time()
    query_embedding = await generate_embedding(query)
    if query_embedding is None:
        print("Failed to generate query embedding")
        return list(range(index.ntotal))
    unique_clusters = set(article.get("cluster", -1) for article in processed_articles)
    centroids = []
    cluster_indices = {}
    for cluster in unique_clusters:
        indices = [i for i, article in enumerate(processed_articles) if article.get("cluster", -1) == cluster]
        if indices:
            try:
                cluster_embeddings = [index.reconstruct(i) for i in indices]
                centroids.append(np.mean(cluster_embeddings, axis=0))
                cluster_indices[cluster] = indices
            except Exception as e:
                print(f"Error computing centroid for cluster {cluster}: {e}")
                continue
    if not centroids:
        print(f"Cluster filtering took {time.time() - start_time:.3f}s")
        return list(range(index.ntotal))
    centroid_index = faiss.IndexFlatL2(dimension)
    centroid_index.add(np.array(centroids, dtype="float32"))
    _, I = centroid_index.search(np.array([query_embedding]), k=1)
    target_cluster = list(unique_clusters)[I[0][0]]
    print(f"Cluster filtering took {time.time() - start_time:.3f}s")
    return cluster_indices.get(target_cluster, [])


def load_precomputed_chunks(articles):
    start_time = time.time()
    chunks = []
    for article in articles:
        if "chunks" in article and "chunk_index" in article:
            try:
                chunks.extend([
                    {"text": chunk, "metadata": {
                        "title": article["title"], "url": article["link"],
                        "date": article["date"], "author": article["author"]
                    }} for chunk in article["chunks"]
                ])
            except Exception as e:
                print(f"Error loading chunks for article {article.get('link', 'unknown')}: {e}")
    print(f"Chunk loading took {time.time() - start_time:.3f}s")
    return chunks

async def retrieve_top_articles(query,index,processed_articles,k=5):
    start_time = time.time()
    filtered_indices = await filter_by_cluster(query,processed_articles,index);
    print(filtered_indices)

    #if nothing comes up from cluster index PLEASE COME UP DAYYYUM I AM PISSED !!
    if not filtered_indices:
        filtered_indices = list(range(index.ntotal)) #TAKE ALL THE ARTICLES

    query_embedding = await generate_embedding(query)
    if query_embedding is None:
        print("Failed to generate embedding of the query ")
        return  []
    filtered_index = faiss.IndexFlatL2(dimension)
    try:
        #fetch already stored indexes created during pipeline
        filtered_index.add(index.reconstruct_n(0, index.ntotal)[filtered_indices])
    except Exception as e:
        print(f"Error building filtered index: {e}")
        return [], []
    D, I = filtered_index.search(np.array([query_embedding]), k=k)
    top_articles = [
        processed_articles[filtered_indices[i]] for i in I[0]
        if filtered_indices[i] < len(processed_articles) and processed_articles[filtered_indices[i]]
    ]
    print(f"Coarse retrieval took {time.time() - start_time:.3f}s")
    return top_articles, D[0]


async def fine_grained_retrieval(query,articles,k=10):
    start = time.time()
    chunk_index = faiss.IndexFlatL2(dimension)
    chunk_texts = []
    chunk_metadata = []
    for article in articles:
        if "chunk_index" in article:
            try:
                article_chunk_index = faiss.read_index(os.path.join(CHUNK_INDEX_DIR,article['chunk_index']))
                chunk_index.merge_from(article_chunk_index,chunk_index.ntotal)
                chunk_texts.extend(article['chunks'])
                chunk_metadata.extend([{
                    "title": article["title"], "url": article["link"],
                    "date": article["date"], "author": article["author"]
                } for _ in article["chunks"]])

            except Exception as e:
                print(f"Error loading chunk index {article['chunk_index']}: {e}")
                continue

    if not chunk_texts:
        print("no chunks available for fine grained retrieval")
    query_embedding = await generate_embedding(query)
    if query_embedding is None:
        print("Failed to generate query embedding for fine-grained retrieval")
        return [], []
    D, I = chunk_index.search(np.array([query_embedding]), k=k)
    top_chunks = [{"text": chunk_texts[i], "metadata": chunk_metadata[i]} for i in I[0] if i < len(chunk_texts)]
    print(f"Fine-grained retrieval took {time.time() - start:.3f}s")
    return top_chunks, D[0]

async def rag_generate(query, chunks, api_key, max_context_length=4000):
    start_time = time.time()
    context = "\n\n".join([chunk["text"] for chunk in chunks])[:max_context_length]
    prompt = f'''You are an AI assistant answering questions based on provided context.
Question: {query}
Context: {context}
Answer in a concise, accurate manner, using the context provided.'''
    try:
        client = Together(api_key=api_key)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        print(f"LLM generation took {time.time() - start_time:.3f}s")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

async def rag_generate_ollama(query, chunks, api_key, max_context_length=4000):
    start_time = time.time()
    context = "\n\n".join([chunk["text"] for chunk in chunks])[:max_context_length]
    prompt = f'''You are an AI assistant answering questions based on provided context.
Question: {query}
Context: {context}
Answer in a concise, accurate manner, using the context provided.'''
    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            think=False,

            messages=[{
                "role": "user",
                "content": prompt
            }],
            options={
                "temperature": 0.5
            }
        )

        print(f"LLM generation took {time.time() - start_time:.3f}s")
        return response.message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None




async def rag_pipeline(query,index,processed_articles,use_fine_grained=True):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{query_hash}.json")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                print(f"Cache hit for query: {query}")
                return json.load(f)
    except Exception as e:
        print(f"Error reading cache file {cache_file}: {e}")
    top_articles, distances = await retrieve_top_articles(query, index, processed_articles, k=5)

    chunks = load_precomputed_chunks(top_articles)

    if use_fine_grained:
        top_chunks, chunk_distances = await fine_grained_retrieval(query, top_articles, k=10)
    else:
        top_chunks = chunks[:10]
        chunk_distances = []

    response = await rag_generate_ollama(query, top_chunks, api_key=None)

    # Save to cache

    print(response)
    result = {
        "query": query,
        "response": response,
        "top_articles": [{
            "title": article["title"],
            "url": article["link"],
            "date": article["date"],
            "author": article["author"]
        } for article in top_articles],
        "top_chunks": [chunk["text"] for chunk in top_chunks]
    }

    print(result)

quuu = "what is kokoda ?"



if __name__ == "__main__":

    asyncio.run(rag_pipeline(quuu,load_stored_faiss_vector(),load_processed_article(),True))





