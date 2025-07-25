
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from backend.services import article_service
from backend.database.connection import get_db
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/")
def get_all_articles(db: Session = Depends(get_db)):
    return article_service.get_all_articles(db)

@router.get("/author/{author}")
def get_articles_by_author(author: str, db: Session = Depends(get_db)):
    return article_service.get_articles_by_author(db, author)

@router.get("/category/{category}")
def get_articles_by_category(category: str, db: Session = Depends(get_db)):
    return article_service.get_articles_by_category(db, category)

@router.get("/search/{query}")
def search_articles(query: str, db: Session = Depends(get_db)):
    return article_service.search_articles(db, query)

@router.get("/highlight/{highlight}")
def get_article_by_highlight(highlight: str, db: Session = Depends(get_db)):
    return article_service.get_article_by_highlight(db, highlight)
