from abc import ABC,abstractmethod
import feedparser




class BaseRssParser(ABC):
    def __init__(self,feed_url):
        self.feed_url = feed_url
        self.feed_data = None

    def fetch_feed(self):
        self.feed_data =feedparser.parse(self.feed_url)

    @abstractmethod
    def parse_feed(self):
        pass



