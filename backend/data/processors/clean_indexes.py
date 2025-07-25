import os
import shutil
from backend.data.config import (
    VECTOR_STORE_PATH,
    PROCESSED_ARTICLES_PATH,
    SEEN_URLS_PATH,
    CHUNK_INDEX_DIR,
    CACHE_DIR,
    RESULTS_PATH,
    GROUPED_PROCESSED_ARTICLES_PATH,
)
from backend.database.connection import SessionLocal
from backend.models.article import Article as DBArticle

def clean_data(confirm=False):
    try:
        files_to_remove = [
            VECTOR_STORE_PATH,
            PROCESSED_ARTICLES_PATH,
            SEEN_URLS_PATH,
            RESULTS_PATH,
            GROUPED_PROCESSED_ARTICLES_PATH
        ]
        directories_to_clear = [
            CHUNK_INDEX_DIR,
            CACHE_DIR
        ]

        print("Files to be removed:")
        for file in files_to_remove:
            print(f"- {file}")
        print("Directories whose contents will be removed:")
        for dir in directories_to_clear:
            print(f"- {dir}")

        if not confirm:
            response = input("Are you sure you want to delete these ? (y/n): ")
            if response.lower() != 'y':
                print("Cleaning aborted by user")
                return False

        for file in files_to_remove:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted file: {file}")
                except Exception as e:
                    print(f"Failed to delete file {file}: {e}")
            else:
                print(f"File not found, skipping: {file}")

        for dir in directories_to_clear:
            if os.path.exists(dir):
                try:
                    for item in os.listdir(dir):
                        item_path = os.path.join(dir, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"Deleted file in directory: {item_path}")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            print(f"Deleted subdirectory: {item_path}")
                except Exception as e:
                    print(f"Failed to clear directory {dir}: {e}")
            else:
                print(f"Directory not found, will create: {dir}")
                os.makedirs(dir, exist_ok=True)

        db = SessionLocal()
        try:
            num_deleted = db.query(DBArticle).delete()
            db.commit()
            print(f"Deleted {num_deleted} articles from the database.")
        except Exception as e:
            db.rollback()
            print(f"Error cleaning database: {e}")
        finally:
            db.close()

        print("Cleaning completed successfully")
        return True

    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False


if __name__ == "__main__":
    clean_data()