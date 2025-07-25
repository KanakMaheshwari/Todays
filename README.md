# Project Overview: Intelligent News Aggregation and Interaction System

This project, "Todays," is designed as a comprehensive system for intelligent news aggregation, processing, and interactive consumption. It addresses the challenge of information overload by providing a streamlined platform for users to access, understand, and engage with news content efficiently. The system aims to transform passive news reading into an active, insightful experience through advanced data processing and artificial intelligence capabilities.

### System Architecture

The system employs a modular, full-stack architecture comprising a robust backend, a responsive frontend, and integrated data storage solutions.

1.  **Backend (Python/FastAPI)**: Serves as the core intelligence and data processing unit. It is responsible for news extraction, content summarization, conversational AI logic, and API endpoint management. FastAPI is utilized for its high performance, asynchronous capabilities, and automatic API documentation generation.
2.  **Frontend (React/Vite)**: Provides the user interface for interacting with the news content. Developed with React, it ensures a dynamic and intuitive user experience, while Vite facilitates rapid development and optimized build processes.
3.  **Data Storage**: A hybrid approach is used for data persistence:
    *   **SQLite**: Manages structured news metadata, article content, and other relational data.
    *   **Vector Store (FAISS)**: Stores high-dimensional embeddings of news articles, enabling efficient similarity searches and facilitating the Retrieval Augmented Generation (RAG) process for conversational AI.

### Key Features and Components

*   **Automated News Extraction and Processing**: The backend incorporates modules for fetching news articles from various sources. These modules are designed to clean, parse, and standardize raw news content for subsequent processing.
*   **AI-Powered Summarization**: Leveraging natural language processing (NLP) techniques, the system generates concise summaries of lengthy articles, allowing users to quickly grasp the main points without reading the full text.
*   **Conversational AI (RAG)**: A significant feature enabling users to interact with news articles through natural language queries. The RAG pipeline retrieves relevant information from the vector store based on user questions and generates coherent, contextually appropriate responses.
*   **Content Categorization**: News articles are automatically categorized to enhance navigability and allow users to filter content based on their interests.
*   **User Interface**: The frontend provides a clean and responsive interface for displaying news articles, category filtering, search functionalities, and the interactive chat feature.

### Technical Stack

*   **Backend**: Python, FastAPI, Uvicorn, SQLAlchemy (or similar ORM), FAISS (for vector storage).
*   **Frontend**: React, TypeScript, Vite, npm/yarn, Tailwind CSS (or similar styling framework).
*   **Database**: SQLite.

### Setup and Execution

To run the Todays project, follow these steps. Ensure you have Python (3.8+) and Node.js (14+) installed on your system.

#### 1. Initialize and Populate the Database (News Pipeline)

First, you need to populate the database with news articles. This step runs the data extraction and processing pipeline.

*   **Purpose**: The `backend/data/processors/process_pipeline.py` script fetches news from configured sources, processes the articles (e.g., cleaning, summarization, generating embeddings), and stores them in the SQLite database and the FAISS vector store. This step is crucial as the frontend and backend rely on this data.
*   **Command**:
    ```bash
    python backend/data/processors/process_pipeline.py
    ```
    This process might take some time depending on the number of articles to fetch and process.

#### 2. Start the Backend Server

Once the database is populated, start the FastAPI backend server.

*   **Purpose**: The backend (`backend/main.py`) exposes the API endpoints that the frontend will consume. It handles requests for news articles, summaries, and conversational AI interactions.
*   **Command**:
    ```bash
    uvicorn backend.main:app --reload
    ```
    The `--reload` flag enables auto-reloading on code changes, which is useful for development. The server will typically run on `http://127.0.0.1:8000`.

#### 3. Start the Frontend Application

Finally, launch the React frontend application.

*   **Purpose**: The frontend provides the user interface for browsing news, interacting with the chat feature, and viewing article details. It communicates with the backend server to fetch data.
*   **Commands**:
    ```bash
    cd frontend/Today's
    npm install # Only run this if you haven't installed dependencies yet
    npm run dev
    ```
    The `npm run dev` command will start the development server, usually opening the application in your web browser at `http://localhost:5173` (or a similar port).

By following these steps, you will have the complete Todays system up and running, with news data populated and both the backend API and frontend UI operational.