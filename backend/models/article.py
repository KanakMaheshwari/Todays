from sqlalchemy import Column, Integer, String, Text
from backend.database.connection import Base

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    link = Column(String, unique=True, index=True)
    author = Column(String)
    date = Column(String)
    content = Column(Text)
    summary = Column(Text)
    category = Column(String)
    img_link = Column(String)
