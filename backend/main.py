import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from backend.api.routes import articles,chat
from backend.api.middleware.cors import add_cors_middleware
from backend.database.connection import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Australian News Aggregator",
    description="An lsAPI for aggregating and interacting with Australian news.",
    version="0.1.0",
)

add_cors_middleware(app)

app.include_router(articles.router, prefix="/api/v1/articles", tags=["Articles"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Australian News Aggregator API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
