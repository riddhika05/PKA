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

# Phi3:mini - Better for code understanding (UPGRADED FROM TINYLLAMA)
MISTRAL_API_URL = "http://localhost:11434/v1"
MISTRAL_MODEL = "phi3:mini"  # Changed from tinyllama - better quality

# Phi3:mini has 4K context window (2x TinyLlama)
MAX_CONTEXT_CHUNKS = 3  # Increased from 2
MAX_SNIPPET_CHARS = 600  # Increased from 300 for better code context
MAX_RESPONSE_TOKENS = 800  # Increased from 300 for full code blocks

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


def query_mistral(prompt: str, max_tokens: int = MAX_RESPONSE_TOKENS) -> str:
    """
    Query Phi3:mini locally via Ollama
    Better for code understanding and generation
    """
    try:
        response = requests.post(
            f"{MISTRAL_API_URL}/chat/completions",
            json={
                "model": MISTRAL_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,  # Lower for more focused code generation
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stream": False,
                "options": {
                    "num_ctx": 4096,  # Phi3's context window
                    "num_thread": 8,
                    "num_predict": max_tokens,
                }
            },
            timeout=300,  # Phi3 is slower than TinyLlama
            stream=False
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Phi3 API error: {response.status_code} - {response.text}"
            )
        
        result = response.json()
        
        # Extract content from response
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0].get('message', {}).get('content', '')
            return content.strip() if content else "No response generated"
        
        return "No response generated"
    
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Phi3. Ensure Ollama is running with: ollama run phi3:mini"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Phi3 took too long to respond."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Phi3 error: {str(e)}"
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
async def ask(question: str, context_limit: int = MAX_CONTEXT_CHUNKS, project_filter: str = None):
    """
    Ask a question about the indexed codebase using RAG with Phi3:mini.
    
    OPTIMIZED FOR CODE GENERATION:
    - Better prompt engineering for code extraction
    - Larger context window (4K tokens vs TinyLlama's 2K)
    - Higher quality model for understanding assignments
    
    Parameters:
    - question: The question to ask
    - context_limit: Number of context chunks to retrieve (default: 3, max: 5)
    - project_filter: Optional filter to search only in a specific project
    """
    
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Enforce maximum context limit
    if context_limit > 5:
        context_limit = 5
        print(f"⚠️ Context limit capped at 5")
    
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
                "sources": [],
                "model": MISTRAL_MODEL
            }
        
        # Build context from search results
        context_chunks = []
        sources = []
        
        for result in search_results:
            # Use larger snippets for better code context
            snippet = result["snippet"][:MAX_SNIPPET_CHARS]
            if len(result["snippet"]) > MAX_SNIPPET_CHARS:
                snippet += "..."
            
            context_chunks.append(snippet)
            sources.append({
                "filename": result["filename"],
                "file_path": result["file_path"],
                "project": result["project_name"],
                "relevance": result["relevance"]
            })
        
        context = "\n\n---SECTION---\n\n".join(context_chunks)
        
        # IMPROVED PROMPT for code generation
        # Detect if question is asking for code
        asking_for_code = any(keyword in question.lower() for keyword in [
            'code', 'implement', 'write', 'program', 'function', 'class', 
            'script', 'assignment', 'solution', 'algorithm'
        ])
        
        if asking_for_code:
            prompt = f"""You are a code assistant. Extract and provide the complete code from the context below.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. If there is code in the context, provide the COMPLETE code with proper formatting
2. Explain what the code does step by step
3. If the code solves an assignment, explain the solution approach
4. Use markdown code blocks with proper language tags
5. DO NOT generate placeholder code - only provide code that exists in the context

ANSWER:"""
        else:
            prompt = f"""Answer the following question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, detailed answer. Include code snippets if relevant using markdown code blocks.

ANSWER:"""
        
        # Check prompt size
        estimated_tokens = len(prompt) // 4
        print(f"📊 Estimated prompt tokens: {estimated_tokens}")
        
        # Query Phi3:mini
        answer = query_mistral(prompt, max_tokens=MAX_RESPONSE_TOKENS)
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_used": len(search_results),
            "sources": sources,
            "model": MISTRAL_MODEL,
            "estimated_prompt_tokens": estimated_tokens
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
    return {
        "status": "ok", 
        "service": "RAG API with Phi3:mini (Code Optimized)",
        "model": MISTRAL_MODEL,
        "max_context_chunks": MAX_CONTEXT_CHUNKS,
        "max_snippet_chars": MAX_SNIPPET_CHARS
    }


@app.get("/model_info")
def model_info():
    """Get current model configuration"""
    return {
        "model": MISTRAL_MODEL,
        "context_window": "~4096 tokens",
        "max_context_chunks": MAX_CONTEXT_CHUNKS,
        "max_snippet_chars": MAX_SNIPPET_CHARS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "optimized_for": "Code generation and understanding",
        "expected_response_time": "15-30 seconds"
    }