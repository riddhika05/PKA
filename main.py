import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import requests
from pathlib import Path

from build_VectorStore import (
    SimpleEmbeddings, 
    WorkspaceSearcher,
    CHROMA_COLLECTION_NAME,
    CHROMA_STORAGE_PATH
)
import chromadb

# CONFIG
WORKSPACE_NAME = "MyRAG_Knowledge_Base"
WORKSPACE_DRIVE = "C:"

# Mistral 7B Local Setup
MISTRAL_API_URL = "http://localhost:11434/v1"  # Ollama default endpoint
MISTRAL_MODEL = "mistral:latest"

# CREATE APP
app = FastAPI()

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components (lazy load)
_embeddings = None
_searcher = None

def get_rag_components():
    """Lazy load RAG components - connects to existing DB"""
    global _embeddings, _searcher
    
    if _embeddings is None:
        _embeddings = SimpleEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Connect to existing Chroma collection (don't create/delete)
        client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
        try:
            collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
            print(f"Connected to existing collection: {CHROMA_COLLECTION_NAME}")
        except Exception as e:
            print(f"Warning: Collection not found. Run build_VectorStore.py first. Error: {e}")
            # Create empty one if it doesn't exist
            collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
        
        # Create searcher with existing collection
        class ExistingVectorStore:
            def __init__(self, collection):
                self.collection = collection
            
            def query(self, query_embedding: List[float], n_results: int = 5):
                return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        
        vector_store = ExistingVectorStore(collection)
        _searcher = WorkspaceSearcher(_embeddings, vector_store)
    
    return _embeddings, _searcher

def query_mistral(prompt: str, max_tokens: int = 256) -> str:
    """Query Mistral 7B locally via Ollama"""
    try:
        response = requests.post(
            f"{MISTRAL_API_URL}/chat/completions",
            json={
                "model": MISTRAL_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stream": False
            },
            timeout=300
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Mistral API error: {response.status_code} - {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Mistral 7B. Ensure Ollama is running: ollama serve"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Mistral error: {str(e)}"
        )

# ENDPOINTS

@app.post("/setup_workspace")
def setup_workspace(workspace_name=WORKSPACE_NAME, drive=WORKSPACE_DRIVE):
    workspace_path = os.path.join(drive, os.sep, workspace_name)

    if os.path.exists(workspace_path):
        return {"status": "success", "message": f"Workspace already exists at {workspace_path}"}
    
    try:
        os.makedirs(workspace_path, exist_ok=True)
        return {"status": "success", "message": f"Workspace created at {workspace_path}"}
    
    except PermissionError:
        raise HTTPException(
            status_code=403, 
            detail=f"Permission denied. Could not create folder at: {workspace_path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    
    try:
        workspace_path = os.path.join(WORKSPACE_DRIVE, os.sep, WORKSPACE_NAME)
        
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if file.filename:
                safe_filename = file.filename.replace(" ", "_")
                file_path = os.path.join(workspace_path, safe_filename)
                
                parent_dir = os.path.dirname(file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "saved_as": safe_filename,
                    "size": len(content),
                    "path": file_path
                })
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(uploaded_files)} file(s)",
            "uploaded_files": uploaded_files,
            "workspace_path": workspace_path
        }
        
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Could not write to workspace directory."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload error: {str(e)}"
        )

@app.post("/ask")
async def ask(question: str, context_limit: int = 5, project_filter: str = None):
    """
    Ask a question about the indexed codebase using RAG with Mistral 7B.
    
    - question: The question to ask
    - context_limit: Number of context chunks to retrieve (default: 5)
    - project_filter: Optional filter to search only in a specific project
    """
    
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get RAG components
        _, searcher = get_rag_components()
        
        # Search for relevant context
        search_results = searcher.search(
            query=question, 
            limit=context_limit, 
            project_filter=project_filter
        )
        
        if not search_results:
            return {
                "status": "success",
                "question": question,
                "answer": "No relevant context found in the knowledge base. Please ensure files have been indexed.",
                "context_used": 0,
                "sources": []
            }
        
        # Build context from search results
        context_chunks = []
        sources = []
        
        for result in search_results:
            context_chunks.append(result["snippet"])
            sources.append({
                "filename": result["filename"],
                "file_path": result["file_path"],
                "project": result["project_name"],
                "relevance": result["relevance"]
            })
        
        context = "\n\n---\n\n".join(context_chunks)
        
        # Build prompt for Mistral
        prompt = f"""You are a helpful code assistant. Answer the following question based on the provided code context.

CONTEXT FROM CODEBASE:
{context}

QUESTION: {question}

Please provide a clear, concise answer based on the context above. If the context doesn't contain relevant information, state that clearly."""
        
        # Query Mistral 7B
        answer = query_mistral(prompt, max_tokens=1024)
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_used": len(search_results),
            "sources": sources
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "RAG API with Mistral 7B"}