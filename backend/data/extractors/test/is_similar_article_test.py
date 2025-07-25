import asyncio
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.cluster import DBSCAN

SIMILARITY_THRESHOLD = 0.90
dimension = 384
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
similarity_search_index = faiss.IndexFlatL2(dimension)
seen_texts = []


def preprocess(article_text):
    tokenized_doc = nlp(article_text.lower())
    tokens = [
        token.lemma_
        for token in tokenized_doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)


async def generate_embedding(article_text):
    return np.array(embedding_model.encode(article_text[:512]), dtype="float32")


async def is_similar_article(article_text, embedding, enable_dbscan=True):
    if similarity_search_index.ntotal > 0:
        D, _ = similarity_search_index.search(np.array([embedding]), k=1)
        distance = D[0][0]
        cosine_sim = 1 - distance / 2
        if cosine_sim > SIMILARITY_THRESHOLD:
            return True

    if enable_dbscan:
        articles_to_compare = seen_texts + [article_text]
        processed = [preprocess(t) for t in articles_to_compare]
        embeddings = embedding_model.encode(processed)
        labels = DBSCAN(eps=0.15, min_samples=2, metric='cosine').fit_predict(embeddings)
        if labels[-1] != -1:
            return True

    return False


async def main():
    articles = [
        """Machine learning is revolutionizing technology by enabling computers to learn from data without explicit programming. Algorithms like neural networks analyze patterns to make predictions or decisions. Applications include image recognition, natural language processing, and autonomous vehicles. Training models requires large datasets and computational power, often using GPUs. Supervised learning uses labeled data, while unsupervised learning finds hidden structures. Reinforcement learning optimizes actions through rewards. Challenges include overfitting, bias in data, and interpretability. Machine learning is driving innovation across industries, from healthcare to finance, transforming how we solve complex problems.""",
        """Machine learning transforms technology, allowing systems to learn from data without direct coding. Neural networks and other algorithms identify patterns for predictions or decisions. It powers image recognition, language processing, and self-driving cars. Training demands big datasets and powerful GPUs. Supervised learning relies on labeled data, while unsupervised learning uncovers hidden patterns. Reinforcement learning improves actions via rewards. Issues like overfitting, biased data, and model interpretability persist. Machine learning fuels advancements in healthcare, finance, and more, reshaping problem-solving in diverse fields.""",
        """Cooking is an art form that combines creativity and science. Techniques like baking, grilling, and sautéing transform raw ingredients into flavorful dishes. Fresh herbs, spices, and quality produce elevate taste. Recipes range from simple stir-fries to complex pastries. Cooking at home fosters healthy eating and family bonding. Modern tools like air fryers and sous-vide machines simplify preparation. Understanding flavor profiles and ingredient pairings is key. Cultural cuisines, like Italian or Thai, offer diverse experiences. Cooking requires practice but rewards with delicious meals and personal satisfaction.""",
        """Soccer is the world’s most popular sport, uniting fans across cultures. Played with 11 players per team, it demands strategy, skill, and teamwork. Matches last 90 minutes, with goals scored by kicking a ball into the opponent’s net. Major tournaments like the FIFA World Cup captivate billions. Players like Messi and Ronaldo inspire with their agility and precision. Training focuses on fitness, ball control, and tactics. Fans create electric atmospheres in stadiums. Soccer promotes physical health and community spirit, making it a global phenomenon.""",
        """Investing in the stock market offers opportunities to grow wealth but carries risks. Stocks represent ownership in companies, and their value fluctuates with market conditions. Diversification across sectors reduces risk. Researching company fundamentals, like earnings and debt, is crucial. Tools like mutual funds and ETFs simplify investing. Market trends, interest rates, and economic data influence stock prices. Long-term strategies, like buy-and-hold, often outperform short-term trading. Financial literacy helps investors make informed decisions. Despite volatility, disciplined investing can build wealth over time."""
        """Investing is like soccer, its an art and an hobbies like to some is cooking or science. many may think it does not require a team like in sports but this is not true, even with the advancement in technology, stock trading is a risky profession and requires a lot of heard work. it is a complex field which involves a lot of earnings or a lot of debt. it requires a lot of computational power, often using GPUs. """

    ]

    similar_count = 0
    processed_articles = []

    print("Processing articles...")
    for i, article_text in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        try:
            embedding = await generate_embedding(article_text)

            is_similar = await is_similar_article(article_text, embedding, enable_dbscan=True)

            if is_similar:
                print(f"Article {i} is similar to a previous article.")
                similar_count += 1
            else:
                print(f"Article {i} is unique.")
                similarity_search_index.add(np.array([embedding]))
                seen_texts.append(article_text)
                processed_articles.append({
                    "article_number": i,
                    "text": article_text[:50] + "...",  # Truncated for brevity
                    "status": "unique"
                })

        except Exception as e:
            print(f"Error processing article {i}: {e}")
            continue

    print("\nResults:")
    print(f"Number of similar articles detected: {similar_count}")
    print(f"Unique articles saved to index: {similarity_search_index.ntotal}")
    print(f"Articles in seen_texts: {len(seen_texts)}")
    print("\nProcessed articles:")
    for article in processed_articles:
        print(f"Article {article['article_number']}: {article['text']} ({article['status']})")


if __name__ == "__main__":
    asyncio.run(main())