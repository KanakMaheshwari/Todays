from .smn_extractor import SydneyMorningNews
from .independent_extractor import IndependentNews
from .gc_extractor import GCNews
from .config import *
from newspaper import Article
from tqdm import tqdm
import json


# def fetch_articles_from_links(all_links):
#     collected_articles = []
#
#     for each_link in tqdm(all_links[:10], desc="Fetching Articles", unit="article"):
#         try:
#             article = Article(each_link['link'])
#             article.download()
#             article.parse()
#
#             if article.text and str(article.text).strip() != "":
#                 fetched_article = article.text
#                 article_detail_dict = {
#                     "title": each_link.get('title', 'No Title'),
#                     "article": fetched_article,
#                     "author": article.authors if article.authors else ["Unknown"],
#                     "date": getattr(article, 'published_date', None) or "Unknown",
#                     "link":each_link['link']
#
#                 }
#                 collected_articles.append(article_detail_dict)
#
#         except Exception as e:
#             print(f"Error processing link {each_link.get('link')}: {e}")
#
#     return json.dumps(collected_articles, indent=5, ensure_ascii=False)
#


def get_all_parsed_article_links_from_rss():
    parsers = [
        GCNews(GC_NEWS)
        # IndependentNews(INDEPENDENT_NEWS),
        # SydneyMorningNews(SYDNEY_MORNING_NEWS)
    ]
    all_articles=[]
    for parser in parsers:
        parser.fetch_feed()
        parsed_articles=parser.parse_feed()
        all_articles.extend(parsed_articles)

    return all_articles


# for a in get_all_parsed_article_links_from_rss():
#     print(a)

# print(beep)
