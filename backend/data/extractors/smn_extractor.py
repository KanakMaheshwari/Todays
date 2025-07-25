from .base_extractor import BaseRssParser

class SydneyMorningNews(BaseRssParser):
    def parse_feed(self):
        articles=[]
        for entry in self.feed_data['entries']:
            articles.append({
                "title":entry.title,
                "link":entry.link
            })
        return articles


# if __name__ == "__main__":
#      lol = SydneyMorningNews("https://www.smh.com.au/rss/national.xml")
#      lol.fetch_feed()
#
#      smn_articles = lol.parse_feed()
#      print(smn_articles)