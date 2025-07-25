from backend.database.repositories.article_repo import ArticleRepository

class NewsService:
    def __init__(self, article_repo: ArticleRepository):
        self.article_repo = article_repo

    def get_all_articles(self):
        return self.article_repo.get_all()
