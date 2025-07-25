from sqlalchemy import Column, Integer, String, Date, Text
from backend.database.connection import Base

class Highlight(Base):
    __tablename__ = "highlights"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True)
    summary = Column(Text)
    top_stories = Column(Text)  # JSON string of top article IDs
