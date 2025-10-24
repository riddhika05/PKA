import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional, Any 
import requests
from pathlib import Path
import shutil
import chromadb
import rag_config 
from rag_config import (
    get_file_reader,
    get_file_splitter,
    get_project_name,
    get_or_create_rag_components,
    WORKSPACE_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_STORAGE_PATH,
    BLACKLIST_FOLDERS,
    FILE_EXTRACTOR
)

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.readers.base import BaseReader
from llama_index.core import SimpleDirectoryReader

MISTRAL_API_URL = "http://localhost:11434/v1"
MISTRAL_MODEL = "phi3:mini"
MAX_CONTEXT_CHUNKS = 3
MAX_SNIPPET_CHARS = 600
MAX_RESPONSE_TOKENS = 2048


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def query_mistral_stream(prompt: str, max_tokens: int = MAX_RESPONSE_TOKENS):
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
            stream=True,
        )

        if response.status_code != 200:
            yield f"data: {json.dumps({'error': f'API error: {response.status_code}'})}\n\n"
            return

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
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
            stream=False,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Phi3 API error: {response.status_code} - {response.text}",
            )

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else "No response generated"

        return "No response generated"

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Phi3. Ensure Ollama is running with: ollama run phi3:mini",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phi3 error: {str(e)}")


@app.post("/setup_workspace")
def setup_workspace():
    if os.path.exists(WORKSPACE_PATH):
        return {
            "status": "success",
            "message": f"Workspace already exists at {WORKSPACE_PATH}",
        }
    try:
        os.makedirs(WORKSPACE_PATH, exist_ok=True)
        return {
            "status": "success",
            "message": f"Workspace created at {WORKSPACE_PATH}",
        }
    except PermissionError:
        raise HTTPException(
            status_code=403, detail=f"Permission denied: {WORKSPACE_PATH}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload_and_index")
async def upload_and_index(
    files: List[UploadFile] = File(...), file_paths: Optional[str] = Form(None)
):
    try:
        index, chroma_collection, embed_model = rag_config.get_or_create_rag_components()
        workspace = Path(WORKSPACE_PATH)

        paths_list = json.loads(file_paths) if file_paths else []
        
        results = []
        total_chunks = 0
        pipeline_cache = {}

        for idx, file in enumerate(files):
            if not file.filename:
                continue

            relative_path = paths_list[idx].replace(" ", "_") if idx < len(paths_list) and paths_list[idx] else file.filename.replace(" ", "_")
            file_path = workspace / relative_path
            
            parts = Path(relative_path).parts
            if any(folder in BLACKLIST_FOLDERS for folder in parts):
                results.append({"filename": file.filename, "status": "skipped", "reason": "Inside blacklisted folder"})
                continue

            file_path.parent.mkdir(parents=True, exist_ok=True)

            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            file_size = len(content)

            reader = get_file_reader(file_path)
            try:
                documents = reader.load_data(file_path)
            except Exception as e:
                 results.append({"filename": file.filename, "saved_as": relative_path, "status": "error", "reason": f"Reader failed: {str(e)}"})
                 continue

            if not documents:
                results.append({"filename": file.filename, "status": "skipped", "reason": "No readable content found", "saved_as": relative_path})
                continue

            file_splitter = get_file_splitter(file_path)
            splitter_name = file_splitter.__class__.__name__

            pipeline_key = (splitter_name, getattr(file_splitter, 'chunk_size', 0), getattr(file_splitter, 'chunk_overlap', 0))
            if pipeline_key not in pipeline_cache:
                pipeline = IngestionPipeline(
                    transformations=[file_splitter, Settings.embed_model],
                    vector_store=ChromaVectorStore(chroma_collection=chroma_collection),
                )
                pipeline_cache[pipeline_key] = pipeline
            else:
                pipeline = pipeline_cache[pipeline_key]

            for doc in documents:
                doc.metadata.update({
                    "file_path": str(file_path.relative_to(workspace)),
                    "absolute_path": str(file_path.resolve()),
                    "project_name": get_project_name(file_path, workspace),
                    "filename": file_path.name,
                    "file_size": file_size,
                    "file_type": file_path.suffix.lower(),
                    "reader": reader.__class__.__name__,
                    "splitter": splitter_name,
                })

            processed_nodes = pipeline.run(documents=documents)
            
            total_chunks += len(processed_nodes)
            
            results.append({
                "filename": file.filename,
                "saved_as": relative_path,
                "size": file_size,
                "path": str(file_path),
                "indexing": {
                    "status": "success", 
                    "chunks_added": len(processed_nodes), 
                    "file_type": file_path.suffix.lower(),
                    "reader": reader.__class__.__name__,
                    "splitter": splitter_name,
                },
            })

        return {
            "status": "success",
            "message": f"Uploaded and indexed {len(results)} file(s)",
            "total_chunks_added": total_chunks,
            "files": results,
            "workspace_path": WORKSPACE_PATH,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/index_workspace")
def index_workspace():
    try:
        index, chroma_collection, _ = rag_config.get_or_create_rag_components()
        workspace = Path(WORKSPACE_PATH)
        if not workspace.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")

        reader = SimpleDirectoryReader(
            input_dir=str(workspace),
            exclude_hidden=True,
            recursive=True,
            file_extractor=rag_config.FILE_EXTRACTOR,
        )

        documents = reader.load_data()
        
        for doc in documents:
            file_path = Path(doc.metadata.get('file_path', ''))
            doc.metadata.update({
                "file_path": str(file_path.relative_to(workspace)) if file_path.is_absolute() else str(file_path),
                "absolute_path": str(file_path.resolve()),
                "project_name": get_project_name(file_path, workspace),
                "filename": file_path.name,
            })
            doc.id_ = str(file_path.resolve())

        for doc in documents:
            index.insert(doc)

        return {
            "status": "success",
            "message": f"Indexed {len(documents)} documents from workspace",
            "total_documents": len(documents),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@app.delete("/delete_file")
def delete_file(file_path: str):
    try:
        index, chroma_collection, _ = rag_config.get_or_create_rag_components()
        workspace = Path(WORKSPACE_PATH)

        full_path = workspace / file_path
        abs_path = str(full_path.resolve())

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        index.delete_ref_doc(ref_doc_id=abs_path, delete_from_docstore=True)
        
        full_path.unlink()

        return {
            "status": "success",
            "message": f"Deleted {file_path}",
            "file_deleted": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.post("/reset_collection")
def reset_collection():
    workspace = Path(WORKSPACE_PATH)
    if workspace.exists():
        for item in workspace.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    try:
        client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        
        client.create_collection(
            name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        rag_config._RAG_COMPONENTS = None
        rag_config.get_or_create_rag_components()

        return {
            "status": "success",
            "message": "Collection and workspace reset successfully",
            "collection_name": CHROMA_COLLECTION_NAME,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


@app.get("/collection_stats")
def collection_stats():
    try:
        index, chroma_collection, _ = rag_config.get_or_create_rag_components()
        
        results = chroma_collection.get(include=["metadatas"])
        total_chunks = len(results["ids"])

        unique_files = set()
        file_types = {}
        
        for metadata in results.get("metadatas", []):
            if metadata:
                abs_path = metadata.get("absolute_path")
                file_type = Path(metadata.get("filename", "")).suffix

                if abs_path:
                    unique_files.add(abs_path)
                
                if file_type:
                    file_types[file_type] = file_types.get(file_type, 0) + 1

        return {
            "status": "success",
            "collection_name": CHROMA_COLLECTION_NAME,
            "total_chunks": total_chunks,
            "unique_files": len(unique_files),
            "file_types": file_types,
            "storage_path": CHROMA_STORAGE_PATH,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.post("/ask")
async def ask(
    question: str,
    context_limit: int = MAX_CONTEXT_CHUNKS,
    project_filter: str = None,
    stream: bool = False,
):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if context_limit > 5:
        context_limit = 5

    try:
        index, chroma_collection, embed_model = rag_config.get_or_create_rag_components()

        retriever = index.as_retriever(
            similarity_top_k=context_limit * 3,
        )

        retrieved_nodes: List[NodeWithScore] = retriever.retrieve(question)

        if project_filter:
            retrieved_nodes = [
                node for node in retrieved_nodes 
                if node.metadata.get("project_name") == project_filter
            ]

        search_results = []
        unique_file_paths = set()
        
        for node in retrieved_nodes:
            md = node.metadata
            file_path = md.get("file_path", "Unknown")
            
            if len(search_results) >= context_limit:
                break
                
            if file_path not in unique_file_paths:
                search_results.append({
                    "rank": len(search_results) + 1,
                    "filename": md.get("filename", "Unknown"),
                    "file_path": file_path,
                    "absolute_path": md.get("absolute_path", ""),
                    "project_name": md.get("project_name", "Unknown"),
                    "relevance": round(node.score, 4) if node.score else 0,
                    "snippet": node.get_text()[:MAX_SNIPPET_CHARS] + ("..." if len(node.get_text()) > MAX_SNIPPET_CHARS else ""),
                    "full_content": node.get_text(),
                })
                unique_file_paths.add(file_path)

        if not search_results:
            if stream:
                async def no_context_stream():
                    yield f"data: {json.dumps({'content': 'No relevant context found in the knowledge base.'})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

                return StreamingResponse(
                    no_context_stream(), media_type="text/event-stream"
                )
            else:
                return {
                    "status": "success",
                    "question": question,
                    "answer": "No relevant context found in the knowledge base.",
                    "context_used": 0,
                    "sources": [],
                    "model": MISTRAL_MODEL,
                }

        file_retrieval_keywords = [
            "file", "path", "source", "location", "give me", "containing function",
            "function", "class", "module", "script", "where is", "find",
        ]
        asking_for_code = any(keyword in question.lower() for keyword in ["code", "implement", "write", "program", "create", "assignment", "solution", "algorithm", "how to"])
        asking_for_filepath_only = (
            any(keyword in question.lower() for keyword in file_retrieval_keywords)
            and not asking_for_code
        )

        if asking_for_filepath_only:
            most_relevant_result = search_results[0]
            file_path = most_relevant_result.get("file_path") or "Unknown"

            answer_parts = [f"The most relevant file is: `{file_path}`"]
            if most_relevant_result.get("filename"):
                answer_parts.append(f"\nFilename: **{most_relevant_result['filename']}**")
            if most_relevant_result.get("project_name"):
                answer_parts.append(f"Project: **{most_relevant_result['project_name']}**")
            answer = "\n".join(answer_parts)

            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "context_used": 1,
                "sources": [
                    {
                        "filename": most_relevant_result.get("filename", "Unknown"),
                        "file_path": file_path,
                        "absolute_path": most_relevant_result.get("absolute_path", ""),
                        "project": most_relevant_result.get("project_name", "Unknown"),
                        "relevance": most_relevant_result.get("relevance", 0),
                    }
                ],
                "model": MISTRAL_MODEL,
            }

        context_chunks = []
        sources = []

        for result in search_results:
            context_chunks.append(result["full_content"])
            sources.append(
                {
                    "filename": result["filename"],
                    "file_path": result["file_path"],
                    "project": result["project_name"],
                    "relevance": result["relevance"],
                }
            )

        context = "\n\n---SECTION---\n\n".join(context_chunks)

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

        if stream:
            async def generate_stream():
                for chunk in query_mistral_stream(
                    prompt, max_tokens=MAX_RESPONSE_TOKENS
                ):
                    yield chunk
                yield f"data: {json.dumps({'sources': sources, 'context_used': len(search_results)})}\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        answer = query_mistral(prompt, max_tokens=MAX_RESPONSE_TOKENS)

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_used": len(search_results),
            "sources": sources,
            "model": MISTRAL_MODEL,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "LlamaIndex RAG API with Dynamic Indexing",
        "model": MISTRAL_MODEL,
    }


@app.get("/list_files")
def list_files():
    try:
        workspace = Path(WORKSPACE_PATH)
        if not workspace.exists():
            return {"status": "success", "files": [], "message": "Workspace not found"}

        files = []
        for root, dirs, filenames in os.walk(workspace):
            dirs[:] = [
                d
                for d in dirs
                if d not in BLACKLIST_FOLDERS
            ]

            for filename in filenames:
                if not filename.startswith("."):
                    file_path = Path(root) / filename
                    rel_path = file_path.relative_to(workspace)
                    files.append(
                        {
                            "path": str(rel_path),
                            "name": filename,
                            "size": file_path.stat().st_size,
                            "extension": file_path.suffix,
                        }
                    )

        return {
            "status": "success",
            "workspace": WORKSPACE_PATH,
            "file_count": len(files),
            "files": files,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    import rag_config 
    rag_config.get_or_create_rag_components() 

    print(f"\n{'='*60}")
    print(f"Starting LlamaIndex RAG API Server")
    print(f"Workspace: {WORKSPACE_PATH}")
    print(f"Model: {MISTRAL_MODEL}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)