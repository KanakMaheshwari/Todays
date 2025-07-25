import hashlib
import asyncio
import json
import faiss
import numpy as np
import spacy
import ollama
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from together import Together
from newspaper import Article
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.data.extractors.rss_main import get_all_parsed_article_links_from_rss
from backend.database.connection import SessionLocal, Base, engine
from backend.models.article import Article as DBArticle

Base.metadata.create_all(bind=engine)


def save_articles_to_db(articles):
    db = SessionLocal()
    try:
        for article_data in articles:
            if article_data:
                existing_article = db.query(DBArticle).filter_by(link=article_data.get("link", "")).first()
                if existing_article:
                    print(f"Skipping duplicate article: {article_data.get('title')} - {article_data.get('link')}")
                    continue

                db_article = DBArticle(
                    title=article_data.get("title", ""),
                    summary=article_data.get("ai_output", {}).get("Summary", ""),
                    category=article_data.get("ai_output", {}).get("Category", ""),
                    link=article_data.get("link", ""),
                    author=", ".join(article_data.get("author", [""])),
                    date=article_data.get("date", ""),
                    content=article_data.get("article", ""),
                    img_link = article_data.get("image","")

                )
                db.add(db_article)
        db.commit()
        print(f"Saved {len(articles)} articles to the database.")
    except Exception as e:
        db.rollback()
        print(f"Error saving articles to database: {e}")
    finally:
        db.close()




from backend.data.config import (
    PROCESSED_ARTICLES_PATH,
    CHUNK_INDEX_DIR,
    VECTOR_STORE_PATH,
    SEEN_URLS_PATH,
    RESULTS_PATH,
    GROUPED_PROCESSED_ARTICLES_PATH,
)


dimension = 384
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
similarity_search_index = None
seen_urls = {}
seen_texts = []
SIMILARITY_THRESHOLD = 0.90


def initialize_vector_store():
    global similarity_search_index
    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading FAISS index...")
        similarity_search_index = faiss.read_index(VECTOR_STORE_PATH)
    else:
        print("FAISS index not found, initializing a new one.")
        similarity_search_index = faiss.IndexFlatL2(dimension)

if os.path.exists(SEEN_URLS_PATH):
    with open(SEEN_URLS_PATH, "r") as f:
        seen_urls = json.load(f)
        seen_texts = list(seen_urls.values())


async def fetch_article(article):
    try:
        current_article = Article(article['link'])
        current_article.download()
        current_article.parse()

        if current_article.text and str(current_article.text).strip() != "":
            article_detail_dict = {
                "title": article.get('title', current_article.title),
                "article": current_article.text,
                "author": current_article.authors if current_article.authors else ["Unknown"],
                "date": getattr(current_article, 'published_date', None) or "Unknown",
                "link": article['link'],
                "image":current_article.top_image

            }

            return article_detail_dict


    except Exception as e:
        print(f"Error processing link {article.get('link')}: {e}")
        return None


def preprocess(article_text):
    tokenized_doc = nlp(article_text.lower())
    tokens = [
        token.lemma_
        for token in tokenized_doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)


async def generate_embedding(article_text):
    return np.array(embedding_model.encode(article_text[:512],batch_size=32), dtype="float32",)

    ### TODO .... KUCH NA, COMING BACK TO YOU AND SLEEP 😴. Good night ❤️


async def is_similar_article(article_text, embedding, enable_dbscan=True):
    if similarity_search_index.ntotal > 0:
        D, _ = similarity_search_index.search(np.array([embedding]), k=1)
        distance = D[0][0]
        cosine_sim = 1 - distance / 2
        if cosine_sim > SIMILARITY_THRESHOLD:
            return True

    if enable_dbscan:
        ### TODO need to optimise this since it looks for in entire historical corpus
        articles_to_compare = seen_texts + [article_text]
        processed = [preprocess(t) for t in articles_to_compare]
        embeddings = embedding_model.encode(processed,batch_size=32)
        labels = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit_predict(embeddings)

        if labels[-1] != -1:
            return True

    return False


def response_to_dic(content):
    replacement = content.replace("\n", ":")
    splitting = replacement.split(':')
    dic = {}
    l = len(splitting)
    for i in range(l):
        if i % 2 == 0:
            continue
        else:
            dic[str(splitting[i - 1]).strip()] = splitting[i]
    return dic

def precompute_chunks(article_text, content_hash):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = text_splitter.split_text(article_text)
    chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=False,batch_size=32)
    chunk_index = faiss.IndexFlatL2(dimension)
    chunk_index.add(np.array(chunk_embeddings, dtype="float32"))
    chunk_index_path = os.path.join(CHUNK_INDEX_DIR, f"chunk_{content_hash}.index")
    faiss.write_index(chunk_index, chunk_index_path)
    return chunks, chunk_index_path

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1,min=2,max=80),
)
async def call_the_llm(article_detail_dict_data):
    prompt = f'''You are news summariser 
     Given the Article text below, extract 
     1.) Categorise into one of these: Sports, Lifestyle, Music , Finance
     2.) Highlight
     3.) Summary (80 words)

     Article:

      {article_detail_dict_data['article'][:4000]}

     Format:
     Category : <one word>
     Highlight : <one line>
     Summary :<80 words>
     '''
    try:
        print("current article length is :", len(prompt))
        client = Together(api_key="0a12e0c577c401ea0e5d79f44fba4fe49b8ef5b865e864c6f6e6965bb756540d")
        response = client.chat.completions.create(

            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.5
        )
        content = response.choices[0].message.content
        response_dict = response_to_dic(content)
        article_detail_dict_data["ai_output"] = response_dict
        # print(content)
        return response_to_dic(content)
    except Exception as e:

        print(f"Error while processing", e)



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=80),
)
async def call_the_ollama(article_detail_dict_data):
    prompt = f'''You are a news summariser 
     Given the Article text below, extract 
     1.) Categorise into one of these: Sports, Lifestyle, Music, Finance
     2.) Highlight
     3.) Summary (80 words)

     Article:

      {article_detail_dict_data['article'][:4000]}

     Format:
     Category : <one word>
     Highlight : <one line>
     Summary : <80 words>
     '''
    try:
        print("Current article prompt length is:", len(prompt))
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
        content = response["message"]["content"]
        response_dict = response_to_dic(content)
        article_detail_dict_data["ai_output"] = response_dict
        return response_dict
    except Exception as e:
        print(f"Error while processing LLM call with Ollama: {e}")
        raise  # Re-raise for tenacity to handle retries


async def process_article(article_detail_dict_data):
    if not article_detail_dict_data or not article_detail_dict_data.get('article'):
        return None

    url = article_detail_dict_data['link']
    content_hash = hashlib.md5(article_detail_dict_data['article'].encode()).hexdigest()

    if url in seen_urls or content_hash in seen_urls.values():
        print("Url already processed : Duplicate")
        return None

    try:
        embedding = await generate_embedding(article_detail_dict_data['article'])
    except Exception as e:
        print(f"Error generating embedding for {url}: {e}")
        return None

    try:
        print("\n Checking for duplicate articles")
        is_similar = await is_similar_article(article_detail_dict_data['article'], embedding, enable_dbscan=True)
        if is_similar:
            print(f"Similar article found {article_detail_dict_data['title']}")
            return None
    except Exception as e:
        print(f"Error checking similarity for {url}: {e}")
        return None

    print("sending call to LLM")
    llm_response = await call_the_llm(article_detail_dict_data)
    # llm_response = await call_the_ollama(article_detail_dict_data)


    # save the embeddings
    if llm_response:
        similarity_search_index.add(np.array([embedding]))
        seen_urls[url] = content_hash
        seen_texts.append(article_detail_dict_data['article'])
        chunks, chunk_index_path = precompute_chunks(article_detail_dict_data['article'], content_hash)
        article_detail_dict_data["ai_output"] = llm_response
        article_detail_dict_data["chunk_index"] = chunk_index_path
        article_detail_dict_data["chunks"] = chunks
    return article_detail_dict_data


def compute_dbscan_clusters(processed_articles):
    if not processed_articles:
        print("No articles to cluster, returning empty labels")
        return []

    valid_embeddings = []
    valid_indices = []

    # Filter valid articles and collect embeddings
    for idx, article in enumerate(processed_articles):
        try:
            if article is not None and "article" in article and article["article"]:
                text = preprocess(article["article"][:512])
                embedding = embedding_model.encode(text, show_progress_bar=False)
                valid_embeddings.append(embedding)
                valid_indices.append(idx)
            else:
                print(f"Skipping invalid article at index {idx}: {article}")
        except Exception as e:
            print(f"Error processing article at index {idx} for embedding: {e}")
            continue

    if not valid_embeddings:
        print("No valid embeddings for clustering, returning default labels")
        return [-1] * len(processed_articles)

    try:
        embeddings = np.array(valid_embeddings)
        labels = DBSCAN(eps=0.6, min_samples=2, metric='cosine').fit_predict(embeddings)

        result_labels = [-1] * len(processed_articles)
        for idx, label in zip(valid_indices, labels):
            result_labels[idx] = label

        return result_labels
    except Exception as e:
        print(f"Error during DBSCAN clustering: {e}")
        return [-1] * len(processed_articles)  # Return default labels on failure

async def process_pipeline(all_articles):
    processed_articles = []
    for article in tqdm(all_articles, desc="Processing articles", unit="fetched_article"):
        try:
            fetched_article = await fetch_article(article)
            processed_article = await process_article(fetched_article)
            if processed_article:
                processed_articles.append(processed_article)
        except Exception as e:
            print(f"Error processing article {article.get('link', 'unknown')}: {e}")
            continue

    print("Processing articles completed")

        # Save state
    faiss.write_index(similarity_search_index, VECTOR_STORE_PATH)
    print("saved similarity FAISS Index")
    with open(SEEN_URLS_PATH, "w") as p:
        json.dump(seen_urls, p)
    print("saved seen urls")


    try:
        cluster_labels = compute_dbscan_clusters(processed_articles)
        labels = [int(label) for label in cluster_labels]
        for article, label in zip(processed_articles, labels):
            if article is not None:
                article["cluster"] = label
            else:
                print("Skipping None article during cluster assignment")
    except Exception as e:
        print(f"Error during DBSCAN clustering or label assignment: {e}")
        for article in processed_articles:
            if article is not None:
                article["cluster"] = -1
    print("Saved DBSCAN clusters")

    with open(PROCESSED_ARTICLES_PATH, "w", encoding="utf-8") as p:
        json.dump(processed_articles, p, ensure_ascii=False, indent=2)
    print(f"Saved {len(processed_articles)} processed articles to {PROCESSED_ARTICLES_PATH}")
    save_articles_to_db(processed_articles)









if __name__ == "__main__":
    initialize_vector_store()
    all_articles = get_all_parsed_article_links_from_rss()
    asyncio.run(process_pipeline(all_articles[:50]))
