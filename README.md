# PKA: Personal Knowledge Assistant (Backend)

An AI-powered Retrieval-Augmented Generation (RAG) backend service that allows users to upload, index, and query their local files and codebases. This service acts as the brain for the **PKA_Frontend** application, providing contextual answers and precise file/code references from your personal workspace.

## 🚀 Project Overview

PKA is designed to turn your local project directory into a searchable knowledge base.
* Users upload files and entire folders into a designated workspace directory on their machine.
* The system indexes the content using a custom, code-focused chunking approach.
* Users can then query the workspace (e.g., "Where is the `handleSendMessage` function defined?").
* The assistant (Misty the Cat 🐈) answers with relevant information, code snippets, and provides the **relative file path** to the source.

## 🔗 Ecosystem Links

This repository is the backend service. To interact with it, you will need the frontend application:

* **PKA Frontend:** https://github.com/riddhika05/PKA_Frontend

## 🧠 Technical Deep Dive: Code-Focused RAG

This project focuses on a lightweight and intuitive approach to RAG, deliberately avoiding heavier dependencies like LangChain and Tree-sitter.

### Custom Indexing & Chunking Logic

The core logic, housed in `build_VectorStore.py`, uses a custom chunking pipeline to maximize the quality of code retrieval:

1.  **Text Files (.md, .txt, etc.):** A simple, intuitive chunking function (`chunk_text_fallback`) splits documents into fixed-size segments (e.g., 1500 characters) with overlap (e.g., 200 characters).
2.  **Code Files (.py, .js, .jsx, .c, etc.):** A more sophisticated function (`chunk_code_by_functions`) utilizes **language-specific regular expressions (regex)** to identify the start of key logical units (functions, classes, components). This ensures that complete, coherent blocks of code are kept intact as single chunks, making the RAG system far more effective for answering code-related questions.

The indexed chunks are stored in a **ChromaDB** vector store, and embeddings are generated using a **Sentence Transformer** model.

### 🛠️ Key API Endpoints

The service is built with **FastAPI** and uses a local Ollama instance running **Phi3:mini** for the LLM component.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/setup_workspace` | `POST` | Creates the local workspace directory on the user's machine. |
| `/upload_and_index` | `POST` | Handles file/folder uploads, saves them to the workspace, and immediately indexes the new content. |
| `/index_workspace` | `POST` | Triggers a re-index of *all* files currently in the workspace folder. |
| `/delete_file` | `DELETE` | Removes a file from both the local workspace directory and the ChromaDB vector store. |
| `/reset_collection`| `POST` | **WARNING:** Deletes all indexed data from the ChromaDB collection. |
| `/collection_stats` | `GET` | Returns statistics on the knowledge base (e.g., total chunks, unique files). |
| `/ask` | `POST` | The main RAG endpoint. Accepts a question, searches the vector store for context, and streams the LLM response. |

## ⚙️ Getting Started (Backend)

### Prerequisites
* Python 3.8+
* A running local LLM instance (e.g., [Ollama](https://ollama.com) running `phi3:mini`).

### Installation
1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server

Start the API server using Uvicorn on port `8000`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
