from backend.database.connection import engine, Base
from backend.models import article, highlight

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created.")
