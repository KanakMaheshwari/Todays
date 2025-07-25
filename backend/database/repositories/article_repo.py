from sqlalchemy.orm import Session
from backend.models.article import Article

class ArticleRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self):
        return self.db.query(Article).all()
