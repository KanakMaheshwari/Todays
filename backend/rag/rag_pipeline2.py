import os
import time
from sqlalchemy.orm import Session
from backend.models.article import Article as DBArticle
from backend.database.connection import SessionLocal

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import ollama

VECTOR_STORE_PATH = 'vector_store/article_text'
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_articles_from_db(db: Session):
    return db.query(DBArticle).all()


def split_chunk(article_text):
    chunk_size = 200
    chunk_overlap = 10
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text("".join(x for x in article_text))
    return splitter.create_documents(chunks)


def initialize_rag():
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Loading existing FAISS index from: {VECTOR_STORE_PATH}")
        return FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)

    db = SessionLocal()
    articles = get_articles_from_db(db)
    db.close()

    if not articles:
        print("No articles found in the DB to build RAG")
        return None

    print("Building FAISS index from articles...")
    articles_text = [article.content for article in articles if article.content]
    splitted = split_chunk(articles_text)
    faiss_index = FAISS.from_documents(splitted, embedding_model)
    faiss_index.save_local(VECTOR_STORE_PATH)
    print(f"FAISS index created and saved to: {VECTOR_STORE_PATH}")
    return faiss_index


def rag_generate_ollama(query, retriever, model_name="deepseek-r1:1.5b", max_context_length=4000):
    try:
        start_time = time.time()

        docs = retriever.get_relevant_documents(query)
        chunks = [doc.page_content for doc in docs]
        context = "\n\n".join(chunks)[:max_context_length]

        prompt = f'''
        You are an AI assistant helping users understand the day's top Australian news highlights.

        Answer the question based ONLY on the provided **articles**, which are curated summaries from multiple reputable Australian news outlets. If the answer is not in the articles, clearly state that.

        Use a clear, helpful, and concise tone. Mention article titles, sources, or frequencies only if directly relevant to the question.

        ---

        Your Question:
        {query}

        ---

        Articles:
        {context}

        ---

        Instructions:
        - Be factual and to the point.
        - Base your answer strictly on the provided articles.
        - If the information is related but not directly answering the question, clarify that and mention what's available.
        - If the articles don’t cover the topic, respond with: “This topic wasn’t covered in today’s news articles.”
        '''

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5},
            think=False

        )

        print(f"LLM generation took {time.time() - start_time:.2f}s")
        return response.message.content

    except Exception as e:
        print(f"Error generating response: {e}")
        return None


retriever_global = initialize_rag().as_retriever()


def query_rag_ollama(question):
    print(f"\n📥 Query: {question}")
    answer = rag_generate_ollama(question, retriever_global)
    print(f"📤 Answer:\n{answer}")


def current_lol_test():
    query_rag_ollama("Who owns Dreamworld?")
    query_rag_ollama("What happened in the latest Aaj Tak article?")


# # Entry point
# if __name__ == "__main__":
#     current_lol_test()
