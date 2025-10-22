import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import requests
from pathlib import Path
import time

from build_VectorStore import (
    SimpleEmbeddings, 
    WorkspaceSearcher,
    WorkspaceIndexer,
    SimpleVectorStore,
    CHROMA_COLLECTION_NAME,
    CHROMA_STORAGE_PATH,
    WHITELIST_EXTENSIONS,
    chunk_code_by_functions,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP
)
import chromadb

# CONFIG
WORKSPACE_NAME = "MyRAG_Knowledge_Base"
WORKSPACE_DRIVE = "C:"
WORKSPACE_PATH = os.path.join(WORKSPACE_DRIVE, os.sep, WORKSPACE_NAME)

# Phi3:mini configuration
MISTRAL_API_URL = "http://localhost:11434/v1"
MISTRAL_MODEL = "phi3:mini"

MAX_CONTEXT_CHUNKS = 3
MAX_SNIPPET_CHARS = 600
MAX_RESPONSE_TOKENS = 2048

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

# Global RAG components
_embeddings = None
_collection = None
_searcher = None


def get_or_create_collection():
    """Get existing collection or create new one if doesn't exist"""
    global _embeddings, _collection, _searcher
    
    if _embeddings is None:
        _embeddings = SimpleEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
        try:
            _collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
            print(f"✅ Connected to existing collection: {CHROMA_COLLECTION_NAME}")
        except Exception:
            _collection = client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Created new collection: {CHROMA_COLLECTION_NAME}")
    
    if _searcher is None:
        class ExistingVectorStore:
            def __init__(self, collection):
                self.collection = collection
            
            def query(self, query_embedding: List[float], n_results: int = 5):
                return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
            
            def add_batch(self, doc_ids, contents, embeddings, metadatas):
                self.collection.add(ids=doc_ids, documents=contents, embeddings=embeddings, metadatas=metadatas)
        
        vector_store = ExistingVectorStore(_collection)
        _searcher = WorkspaceSearcher(_embeddings, vector_store)
    
    return _embeddings, _collection, _searcher


def index_single_file(file_path: Path, embeddings, collection) -> dict:
    """Index a single file and add its chunks to the collection"""
    ext = file_path.suffix.lower()
    
    if ext not in WHITELIST_EXTENSIONS:
        return {"status": "skipped", "reason": "file type not whitelisted"}
    
    try:
        size = file_path.stat().st_size
        if size > 2_000_000:  # 2MB limit
            return {"status": "skipped", "reason": "file too large"}
        
        # Read file content
        if ext == ".pdf":
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                return {"status": "error", "reason": f"PDF read error: {str(e)}"}
        else:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                return {"status": "error", "reason": f"File read error: {str(e)}"}
        
        if not text.strip():
            return {"status": "skipped", "reason": "empty file"}
        
        # Chunk the text
        chunks = chunk_code_by_functions(text, ext, max_chunk_chars=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP)
        
        # Prepare batch data
        batch_ids = []
        batch_contents = []
        batch_embeddings = []
        batch_metadatas = []
        
        abs_path = str(file_path.resolve())
        
        for idx, (chunk_text, start_char, end_char) in enumerate(chunks):
            chunk_text_clean = chunk_text.strip()
            if not chunk_text_clean:
                continue
            
            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path.relative_to(WORKSPACE_PATH)) if WORKSPACE_PATH in str(file_path) else str(file_path),
                "absolute_path": abs_path,
                "project_name": file_path.parts[-2] if len(file_path.parts) > 1 else "root",
                "file_type": ext,
                "file_size": int(size),
                "chunk_index": int(idx),
                "chunk_char_start": int(start_char),
                "chunk_char_end": int(end_char),
                "chunk_size": len(chunk_text_clean)
            }
            
            doc_id = f"{abs_path}::chunk_{idx}"
            
            try:
                emb = embeddings.embed(chunk_text_clean)
            except Exception as e:
                continue
            
            batch_ids.append(doc_id)
            batch_contents.append(chunk_text_clean)
            batch_embeddings.append(emb)
            batch_metadatas.append(metadata)
        
        # Add to collection
        if batch_ids:
            collection.add(
                ids=batch_ids,
                documents=batch_contents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        return {
            "status": "success",
            "chunks_added": len(batch_ids),
            "file_size": size
        }
    
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def delete_file_from_collection(file_path: Path, collection) -> dict:
    """Delete all chunks of a file from the collection"""
    abs_path = str(file_path.resolve())
    
    try:
        # Get all document IDs for this file
        results = collection.get()
        ids_to_delete = [id for id in results['ids'] if id.startswith(f"{abs_path}::")]
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return {"status": "success", "chunks_deleted": len(ids_to_delete)}
        else:
            return {"status": "success", "chunks_deleted": 0, "message": "File not found in collection"}
    
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def query_mistral_stream(prompt: str, max_tokens: int = MAX_RESPONSE_TOKENS):
    """Query Phi3:mini with streaming"""
    try:
        response = requests.post(
            f"{MISTRAL_API_URL}/chat/completions",
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stream": True,
            },
            timeout=300,
            stream=True
        )
        
        if response.status_code != 200:
            yield f"data: {json.dumps({'error': f'API error: {response.status_code}'})}\n\n"
            return
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    except requests.exceptions.ConnectionError:
        yield f"data: {json.dumps({'error': 'Cannot connect to Phi3. Ensure Ollama is running'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def query_mistral(prompt: str, max_tokens: int = MAX_RESPONSE_TOKENS) -> str:
    """Query Phi3:mini locally via Ollama (non-streaming)"""
    try:
        response = requests.post(
            f"{MISTRAL_API_URL}/chat/completions",
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=300,
            stream=False
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Phi3 API error: {response.status_code} - {response.text}"
            )
        
        result = response.json()
        
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
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phi3 error: {str(e)}")


# ==================== API ENDPOINTS ====================

@app.post("/setup_workspace")
def setup_workspace():
    """Create workspace directory if it doesn't exist"""
    if os.path.exists(WORKSPACE_PATH):
        return {"status": "success", "message": f"Workspace already exists at {WORKSPACE_PATH}"}
    
    try:
        os.makedirs(WORKSPACE_PATH, exist_ok=True)
        return {"status": "success", "message": f"Workspace created at {WORKSPACE_PATH}"}
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {WORKSPACE_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload_and_index")
async def upload_and_index(
    files: List[UploadFile] = File(...),
    file_paths: Optional[str] = Form(None)
):
    """Upload files and automatically index them into the vector store"""
    try:
        if not os.path.exists(WORKSPACE_PATH):
            os.makedirs(WORKSPACE_PATH, exist_ok=True)
        
        embeddings, collection, searcher = get_or_create_collection()
        
        paths_list = []
        if file_paths:
            try:
                paths_list = json.loads(file_paths)
                print(f"📂 Folder upload detected with {len(paths_list)} paths")
            except Exception as e:
                print(f"⚠️ Could not parse file_paths: {e}")
                paths_list = []
        
        results = []
        total_chunks = 0
        
        for idx, file in enumerate(files):
            if not file.filename:
                continue
            
            if idx < len(paths_list) and paths_list[idx]:
                relative_path = paths_list[idx].replace(" ", "_")
            else:
                relative_path = file.filename.replace(" ", "_")
            
            file_path = Path(WORKSPACE_PATH) / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            index_result = index_single_file(file_path, embeddings, collection)
            
            results.append({
                "filename": file.filename,
                "saved_as": relative_path,
                "size": len(content),
                "path": str(file_path),
                "indexing": index_result
            })
            
            if index_result["status"] == "success":
                total_chunks += index_result.get("chunks_added", 0)
        
        return {
            "status": "success",
            "message": f"Uploaded and indexed {len(results)} file(s)",
            "total_chunks_added": total_chunks,
            "files": results,
            "workspace_path": WORKSPACE_PATH
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/index_workspace")
def index_workspace():
    """Index all files in the workspace directory"""
    try:
        workspace = Path(WORKSPACE_PATH)
        if not workspace.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        embeddings, collection, searcher = get_or_create_collection()
        
        all_files = []
        for root, dirs, files in os.walk(workspace):
            dirs[:] = [d for d in dirs if d not in {"node_modules", "__pycache__", "venv", ".git"}]
            
            for file in files:
                if not file.startswith("."):
                    all_files.append(Path(root) / file)
        
        results = []
        total_chunks = 0
        
        for file_path in all_files:
            index_result = index_single_file(file_path, embeddings, collection)
            
            if index_result["status"] == "success":
                total_chunks += index_result.get("chunks_added", 0)
            
            results.append({
                "file": str(file_path.relative_to(workspace)),
                "result": index_result
            })
        
        return {
            "status": "success",
            "message": f"Indexed {len(all_files)} files",
            "total_chunks_added": total_chunks,
            "files": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@app.delete("/delete_file")
def delete_file(file_path: str):
    """Delete a file from both filesystem and vector store"""
    try:
        embeddings, collection, searcher = get_or_create_collection()
        
        full_path = Path(WORKSPACE_PATH) / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        delete_result = delete_file_from_collection(full_path, collection)
        full_path.unlink()
        
        return {
            "status": "success",
            "message": f"Deleted {file_path}",
            "vector_store": delete_result,
            "file_deleted": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.post("/reset_collection")
def reset_collection():
    """Delete and recreate the entire collection"""
    global _collection, _searcher
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
        
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        
        _collection = client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        _searcher = None
        
        return {
            "status": "success",
            "message": "Collection reset successfully",
            "collection_name": CHROMA_COLLECTION_NAME
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


@app.get("/collection_stats")
def collection_stats():
    """Get statistics about the current collection"""
    try:
        embeddings, collection, searcher = get_or_create_collection()
        
        results = collection.get()
        total_chunks = len(results['ids'])
        
        unique_files = set()
        for metadata in results.get('metadatas', []):
            if metadata and 'absolute_path' in metadata:
                unique_files.add(metadata['absolute_path'])
        
        file_types = {}
        for metadata in results.get('metadatas', []):
            if metadata and 'file_type' in metadata:
                ft = metadata['file_type']
                file_types[ft] = file_types.get(ft, 0) + 1
        
        return {
            "status": "success",
            "collection_name": CHROMA_COLLECTION_NAME,
            "total_chunks": total_chunks,
            "unique_files": len(unique_files),
            "file_types": file_types,
            "storage_path": CHROMA_STORAGE_PATH
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.post("/ask")
async def ask(question: str, context_limit: int = MAX_CONTEXT_CHUNKS, project_filter: str = None, stream: bool = False):
    """Ask a question about the indexed codebase using RAG with Phi3:mini.
    Includes logic for detecting and returning file paths for specific file/function queries.
    """

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if context_limit > 5:
        context_limit = 5

    try:
        embeddings, collection, searcher = get_or_create_collection()
        
        # --- 1. Intent Detection for File Path Retrieval ---
        # Checks if the user is explicitly asking for a file/path/source
        file_retrieval_keywords = [
            'file', 'path', 'source', 'location', 'give me', 'containing function', 
            'function', 'class', 'module', 'script'
        ]
        
        asking_for_code = any(keyword in question.lower() for keyword in [
            'code', 'implement', 'write', 'program', 'function', 'class', 
            'script', 'assignment', 'solution', 'algorithm'
        ])
        
        asking_for_filepath_only = any(
            keyword in question.lower() for keyword in ['file', 'path', 'source', 'location']
        ) and not asking_for_code
        
        # --- 2. Perform RAG Search (Always necessary to find context) ---
        search_results = searcher.search(
            query=question, 
            limit=context_limit, 
            project_filter=project_filter
        )
        
        print(f"\n--- RAG Search Results for Question: '{question}' ---")
        if not search_results:
            print("No results found.")
        else:
            for i, result in enumerate(search_results):
                print(f"[{i+1}] Relevance: {result['relevance']:.4f} | File: {result['file_path']}")
        print("----------------------------------------------------------")

        # --- 3. Handle No Context Found ---
        if not search_results:
            if stream:
                async def no_context_stream():
                    yield f"data: {json.dumps({'content': 'No relevant context found in the knowledge base.'})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
                return StreamingResponse(no_context_stream(), media_type="text/event-stream")
            else:
                return {
                    "status": "success",
                    "question": question,
                    "answer": "No relevant context found in the knowledge base.",
                    "context_used": 0,
                    "sources": [],
                    "model": MISTRAL_MODEL
                }

        # --- 4. Handle Explicit File Path Request ---
        if asking_for_filepath_only:
            # Get the single most relevant file path
            most_relevant_result = search_results[0]
            file_path = most_relevant_result['file_path']
            
            # Return the file path directly without calling the LLM
            return {
                "status": "success",
                "question": question,
                "answer": f"The most relevant file is: `{file_path}`",
                "context_used": 1,
                "sources": [{
                    "filename": most_relevant_result["filename"],
                    "file_path": file_path,
                    "project": most_relevant_result["project_name"],
                    "relevance": most_relevant_result["relevance"]
                }],
                "model": MISTRAL_MODEL
            }
        
        # --- 5. Prepare Context for LLM (Standard RAG) ---
        context_chunks = []
        sources = []
        
        for result in search_results:
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
        
        # Determine prompt type (Code Generation vs. General Answer)
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
            prompt = f"""Answer the following question based on the provided context. If context is insufficient say so.
            
CONTEXT:
{context}

QUESTION: {question}

Provide a clear, detailed answer. Include code snippets if relevant using markdown code blocks.

ANSWER:"""
        
        # --- 6. LLM Response ---
        # Streaming response
        if stream:
            async def generate_stream():
                for chunk in query_mistral_stream(prompt, max_tokens=MAX_RESPONSE_TOKENS):
                    yield chunk
                # Send sources at the end
                yield f"data: {json.dumps({'sources': sources, 'context_used': len(search_results)})}\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        # Non-streaming response
        answer = query_mistral(prompt, max_tokens=MAX_RESPONSE_TOKENS)
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_used": len(search_results),
            "sources": sources,
            "model": MISTRAL_MODEL
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "service": "RAG API with Dynamic Indexing",
        "model": MISTRAL_MODEL
    }


@app.get("/list_files")
def list_files():
    """List all files in the workspace"""
    try:
        workspace = Path(WORKSPACE_PATH)
        if not workspace.exists():
            return {"status": "success", "files": [], "message": "Workspace not found"}
        
        files = []
        for root, dirs, filenames in os.walk(workspace):
            dirs[:] = [d for d in dirs if d not in {"node_modules", "__pycache__", "venv", ".git"}]
            
            for filename in filenames:
                if not filename.startswith("."):
                    file_path = Path(root) / filename
                    rel_path = file_path.relative_to(workspace)
                    files.append({
                        "path": str(rel_path),
                        "name": filename,
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix
                    })
        
        return {
            "status": "success",
            "workspace": WORKSPACE_PATH,
            "file_count": len(files),
            "files": files
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")