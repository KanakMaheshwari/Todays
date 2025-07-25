from .base_extractor import BaseRssParser

class IndependentNews(BaseRssParser):
    def parse_feed(self):
        articles=[]
        for entry in self.feed_data['entries']:
            articles.append({
                "title":entry.title,
                "link": entry.link
            })
        return articles

#
# if __name__ == "__main__":
#      lol = IndependentNews("https://feeds.feedburner.com/IndependentAustralia")
#      lol.fetch_feed()
#
#      independent_articles = lol.parse_feed()
#      print(independent_articles)
