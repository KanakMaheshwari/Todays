
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.models.article import Article
from backend.database.connection import SessionLocal, Base, engine



def get_all_articles(db: Session):
    return db.query(Article).all()

def get_articles_by_author(db: Session, author: str):
    return db.query(Article).filter(Article.author.contains(author)).all()

def get_articles_by_category(db: Session, category: str):
    return db.query(Article).filter(Article.category.ilike(f"%{category}%")).all()

def search_articles(db: Session, query: str):
    return db.query(Article).filter(
        (Article.title.ilike(f"%{query}%")) | (Article.content.ilike(f"%{query}%"))
    ).all()

def get_article_by_highlight(db: Session, highlight: str):
    return db.query(Article).filter(Article.summary.ilike(f"%{highlight}%")).all()
