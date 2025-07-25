from .base_extractor import BaseRssParser

class GCNews(BaseRssParser):
    def parse_feed(self):
        articles=[]
        for entry in self.feed_data['entries']:
            articles.append({
                "title":entry.title,
                "link":entry.link
            })
        return  articles






# if __name__ == "__main__":
#      lol = ABCNews("https://www.news.com.au/content-feeds/latest-news-national/")
#      lol.fetch_feed()
#
#      abc_articles = lol.parse_feed()
#      print(abc_articles)




