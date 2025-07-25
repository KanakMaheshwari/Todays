from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    return " ".join(tokens)


def compute_dbscan_clusters(processed_articles, eps=0.4, min_samples=1):
    if not processed_articles:
        return []
    embeddings = [embedding_model.encode(preprocess(article["article"])[:512], show_progress_bar=False)
                  for article in processed_articles]

    sims = cosine_similarity(embeddings)
    print("\nCosine Similarity Matrix:")
    print(np.round(sims, 2))

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(np.array(embeddings))
    return labels.tolist()


def compute_dbscan_clusters_test():
    processed_articles = [
        {
            "article": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence."},
        {
            "article": "Machine learning is a type of AI that allows software applications to become more accurate at predicting outcomes without being explicitly programmed."},
        {
            "article": "Cooking is both an art and a science. It involves the preparation of food using heat and combining ingredients in various ways."},
        {
            "article": "Cooking requires a deep understanding of ingredients, flavor, and timing. Many cultural traditions use cooking as a form of storytelling."}
    ]

    labels = compute_dbscan_clusters(processed_articles, eps=0.6, min_samples=1)
    print("\nDBSCAN Cluster Labels:")
    print(labels)


if __name__ == "__main__":
    compute_dbscan_clusters_test()
