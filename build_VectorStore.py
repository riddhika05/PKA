#!/usr/bin/env python3

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
import time


try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("⚠️  PyPDF2 not installed. PDF text extraction disabled.")
    print("   Run: pip install PyPDF2\n")


WORKSPACE_PATH = r"C:\MyRAG_Knowledge_Base"

WHITELIST_EXTENSIONS = {
    ".py", ".cs", ".js", ".ts", ".java", ".go",
    ".html", ".css", ".md", ".txt", ".json",
    ".csproj", ".sln", ".cshtml", ".config", ".xml",
    ".pdf", ".jsx", ".tsx", ".md", ".ppt", ".pptx"
}

BLACKLIST_FOLDERS = {
    "node_modules", "__pycache__", "venv", "env", ".git", ".idea", ".vscode",
    "dist", "build", "target", ".next", ".nuxt",
    "bin", "obj", "packages", "Migrations" 
}
MAX_FILE_SIZE_BYTES = 2_000_000

# OPTIMIZED FOR PHI3:MINI - Balanced chunks for code
# Phi3 has ~4096 token context window
CHUNK_SIZE_CHARS = 1500  # Balanced size for code blocks
CHUNK_OVERLAP = 200       # Good overlap for context continuity

CHROMA_COLLECTION_NAME = "documents"
CHROMA_STORAGE_PATH = "rag_chroma_db"


class SimpleEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded.\n")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


class SimpleVectorStore:
    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME, reset: bool = True):
        self.client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
        
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Deleted old Chroma collection '{collection_name}'.")
            except Exception:
                pass 
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new Chroma collection '{collection_name}' in '{CHROMA_STORAGE_PATH}'.\n")

    def add_batch(self, doc_ids: List[str], contents: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """Add multiple documents at once for better performance"""
        try:
            self.collection.add(
                ids=doc_ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Error adding batch: {e}")
            return False

    def add(self, doc_id: str, content: str, embedding: List[float], metadata: Dict):
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def query(self, query_embedding: List[float], n_results: int = 5):
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)


def files_within_depth(root: Path) -> List[Path]:
    root = root.resolve()
    print(f"\n🔍 SCANNING: {root}\n")
    
    if not root.exists():
        print(f"❌ Root path does not exist!")
        return []
    
    results = []
    extensions = {}
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in BLACKLIST_FOLDERS]
        
        current_dir_name = Path(dirpath).name
        if current_dir_name in BLACKLIST_FOLDERS:
             continue
        
        for filename in filenames:
            if filename.startswith("."):
                continue
            
            filepath = Path(dirpath) / filename
            results.append(filepath)
            
            ext = filepath.suffix.lower() if filepath.suffix else "[no extension]"
            extensions[ext] = extensions.get(ext, 0) + 1
    
    print(f"✅ Found {len(results)} total files from os.walk()\n")
    
    print("FILE TYPES FOUND:")
    for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
        whitelisted = "✓" if ext in WHITELIST_EXTENSIONS else "✗"
        print(f"  [{whitelisted}] {ext}: {count} files")
    print()
    
    return results


def chunk_text_fallback(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str,int,int]]:
    """
    OPTIMIZED FOR PHI3:MINI:
    Balanced chunks that can hold complete code functions/blocks
    """
    chunks = []
    text_len = len(text)
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks

FUNCTION_PATTERNS = {
    ".py": re.compile(r'^\s*(def|class)\s+[A-Za-z0-9_]+\s*[\(:]', re.MULTILINE),
    ".js": re.compile(r'^\s*(function\s+[A-Za-z0-9_]+|\b[A-Za-z0-9_]+\s*=\s*function\b|\b[A-Za-z0-9_]+\s*=\s*\(.*\)\s*=>)', re.MULTILINE),
    ".ts": re.compile(r'^\s*(function\s+[A-Za-z0-9_]+|\b[A-Za-z0-9_]+\s*=\s*function\b|\b[A-Za-z0-9_]+\s*=\s*\(.*\)\s*=>)', re.MULTILINE),
    ".java": re.compile(r'^\s*(public|private|protected)?\s*(static)?\s*[A-Za-z0-9_<>\[\]]+\s+[A-Za-z0-9_]+\s*\(', re.MULTILINE),
    ".go": re.compile(r'^\s*func\s+[A-Za-z0-9_]+\s*\(', re.MULTILINE),
    ".cs": re.compile(r'^\s*(public|private|protected|internal)?\s*(static)?\s*[A-Za-z0-9_<>\[\]]+\s+[A-Za-z0-9_]+\s*\(', re.MULTILINE),
    ".jsx": re.compile(r'^\s*(const\s+[A-Za-z0-9_]+\s*=\s*\(.*?\)\s*=>|function\s+[A-Za-z0-9_]+|\b[A-Za-z0-9_]+\s*=\s*function\b|\bconst\s+[A-Za-z0-9_]+\s*=\s*function\b)', re.MULTILINE),
    ".tsx": re.compile(r'^\s*(const\s+[A-Za-z0-9_]+\s*=\s*\(.*?\)\s*=>|function\s+[A-Za-z0-9_]+|\b[A-Za-z0-9_]+\s*=\s*function\b|\bconst\s+[A-Za-z0-9_]+\s*=\s*function\b)', re.MULTILINE),
}

def chunk_code_by_functions(text: str, ext: str, max_chunk_chars: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str,int,int]]:
    """
    OPTIMIZED FOR PHI3:MINI - Better code block preservation
    """
    pattern = FUNCTION_PATTERNS.get(ext)
    if not pattern:
        return chunk_text_fallback(text, max_chunk_chars, overlap)

    matches = list(pattern.finditer(text))
    if not matches:
        return chunk_text_fallback(text, max_chunk_chars, overlap)

    starts = [m.start() for m in matches]
    starts.append(len(text))
    chunks = []
    for i in range(len(starts) - 1):
        s = starts[i]
        e = starts[i+1]
        chunk_text = text[s:e].rstrip()
        if len(chunk_text) > max_chunk_chars:
            subchunks = chunk_text_fallback(chunk_text, max_chunk_chars, overlap)
            for sub, sub_s, sub_e in subchunks:
                global_s = s + sub_s
                global_e = s + sub_e
                chunks.append((sub, global_s, global_e))
        else:
            chunks.append((chunk_text, s, e))
    
    # Merge small chunks
    merged = []
    for chunk, s, e in chunks:
        if not merged:
            merged.append((chunk, s, e))
            continue
        prev_chunk, prev_s, prev_e = merged[-1]
        if len(chunk) < 200 and (len(prev_chunk) + len(chunk)) < max_chunk_chars:
            merged[-1] = (prev_chunk + "\n\n" + chunk, prev_s, e)
        else:
            merged.append((chunk, s, e))
    
    final = []
    for chunk_text, s, e in merged:
        if len(chunk_text) > max_chunk_chars:
            subs = chunk_text_fallback(chunk_text, max_chunk_chars, overlap)
            for sub, sub_s, sub_e in subs:
                final.append((sub, s + sub_s, s + sub_e))
        else:
            final.append((chunk_text, s, e))
    return final

class WorkspaceIndexer:
    def __init__(self, workspace_root: str, embeddings: SimpleEmbeddings, vector_store: SimpleVectorStore):
        self.workspace_root = Path(workspace_root).resolve()
        self.embeddings = embeddings
        self.vector_store = vector_store

    def get_project_name(self, file_path: Path) -> str:
        try:
            rel = file_path.relative_to(self.workspace_root)
            parts = rel.parts
            if len(parts) > 0:
                return parts[0]
            return "root"
        except Exception:
            return "unknown"

    def index(self) -> Dict[str,int]:
        if not self.workspace_root.exists():
            raise FileNotFoundError(f"Workspace root not found: {self.workspace_root}")

        all_files = files_within_depth(self.workspace_root)

        file_count = 0
        chunk_count = 0
        start_time = time.time()

        print("=" * 60)
        print("INDEXING FILES (Optimized for Phi3:mini - Code Focus)")
        print(f"Chunk size: {CHUNK_SIZE_CHARS} chars (balanced for code blocks)")
        print("=" * 60 + "\n")

        batch_ids = []
        batch_contents = []
        batch_embeddings = []
        batch_metadatas = []
        batch_size = 50

        for file_path in all_files:
            ext = file_path.suffix.lower()
            
            if ext not in WHITELIST_EXTENSIONS:
                continue

            try:
                size = file_path.stat().st_size
            except Exception:
                continue
            
            if size > MAX_FILE_SIZE_BYTES:
                print(f"⊘ Skipping {file_path.name} (too large: {size/1024/1024:.1f} MB)")
                continue

            if ext == ".pdf":
                if not HAS_PDF:
                     print(f"⊘ Skipping {file_path.name} (PDF parsing library not installed)")
                     continue
                try:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    # Log PDF content for debugging
                    print(f"📄 PDF extracted: {file_path.name} ({len(text)} chars)")
                    if len(text) > 100:
                        print(f"   First 100 chars: {text[:100]}")
                        
                except Exception as e:
                    print(f"❌ Error reading PDF {file_path.name}: {e}")
                    continue
            else:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"❌ Error reading {file_path.name}: {e}")
                    continue

            if not text.strip():
                continue

            # Use optimized chunking
            chunks = chunk_code_by_functions(text, ext, max_chunk_chars=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP)

            project_name = self.get_project_name(file_path)
            rel_path = str(file_path.relative_to(self.workspace_root))
            abs_path = str(file_path.resolve())

            for idx, (chunk_text, start_char, end_char) in enumerate(chunks):
                chunk_text_clean = chunk_text.strip()
                if not chunk_text_clean:
                    continue

                metadata = {
                    "filename": file_path.name,
                    "file_path": rel_path,
                    "absolute_path": abs_path,
                    "project_name": project_name,
                    "file_type": ext,
                    "file_size": size,
                    "chunk_index": idx,
                    "chunk_char_start": int(start_char),
                    "chunk_char_end": int(end_char),
                    "chunk_size": len(chunk_text_clean)
                }

                doc_id = f"{abs_path}::chunk_{idx}"

                try:
                    emb = self.embeddings.embed(chunk_text_clean)
                except Exception as e:
                    print(f"❌ Embedding error for {doc_id}: {e}")
                    continue

                # Add to batch
                batch_ids.append(doc_id)
                batch_contents.append(chunk_text_clean)
                batch_embeddings.append(emb)
                batch_metadatas.append(metadata)

                # Flush batch when it reaches batch_size
                if len(batch_ids) >= batch_size:
                    self.vector_store.add_batch(batch_ids, batch_contents, batch_embeddings, batch_metadatas)
                    chunk_count += len(batch_ids)
                    batch_ids = []
                    batch_contents = []
                    batch_embeddings = []
                    batch_metadatas = []

            file_count += 1
            print(f"✅ Indexed: {rel_path} ({len(chunks)} chunks)")

        # Flush remaining batch
        if batch_ids:
            self.vector_store.add_batch(batch_ids, batch_contents, batch_embeddings, batch_metadatas)
            chunk_count += len(batch_ids)

        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"Indexing complete in {elapsed:.1f}s")
        print(f"Files indexed: {file_count}")
        print(f"Total chunks indexed: {chunk_count}")
        print(f"Average chunk size: {CHUNK_SIZE_CHARS} chars (optimized for Phi3:mini)")
        print("="*60 + "\n")
        return {"files": file_count, "chunks": chunk_count}

class WorkspaceSearcher:
    def __init__(self, embeddings: SimpleEmbeddings, vector_store: SimpleVectorStore):
        self.embeddings = embeddings
        self.vector_store = vector_store

    def search(self, query: str, limit: int = 5, project_filter: str = None):
        """
        OPTIMIZED FOR PHI3:MINI - Returns larger snippets for code context
        """
        q_emb = self.embeddings.embed(query)
        res = self.vector_store.query(q_emb, n_results=limit * 3)

        formatted = []
        ids = res.get("ids", [[]])[0] if isinstance(res.get("ids"), list) else res.get("ids")[0]
        docs = res.get("documents", [[]])[0] if isinstance(res.get("documents"), list) else res.get("documents")[0]
        metadatas = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas"), list) else res.get("metadatas")[0]
        distances = res.get("distances", [[]])[0] if isinstance(res.get("distances"), list) else res.get("distances")[0]

        for i, doc_id in enumerate(ids):
            if i >= len(metadatas):
                break
            md = metadatas[i]
            if project_filter and md.get("project_name") != project_filter:
                continue
            doc_content = docs[i] if i < len(docs) else ""
            distance = distances[i] if i < len(distances) else 1.0
            
            # Return larger snippets for Phi3:mini (600 chars for better code context)
            snippet_length = 600  # Increased from 300 for code blocks
            formatted.append({
                "rank": len(formatted) + 1,
                "doc_id": doc_id,
                "filename": md.get("filename"),
                "file_path": md.get("file_path"),
                "project_name": md.get("project_name"),
                "snippet": doc_content[:snippet_length] + ("..." if len(doc_content) > snippet_length else ""),
                "full_content": doc_content,  # Include full chunk for code extraction
                "relevance": round(1 - distance, 4)
            })
            if len(formatted) >= limit:
                break
        return formatted


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VECTOR STORE BUILDER - OPTIMIZED FOR PHI3:MINI")
    print("="*60)
    print(f"Chunk size: {CHUNK_SIZE_CHARS} chars (balanced for code)")
    print(f"Chunk overlap: {CHUNK_OVERLAP} chars")
    print(f"Target: Complete code blocks for Phi3's 4K context")
    print("="*60 + "\n")
    
    root = Path(WORKSPACE_PATH)
    embeddings = SimpleEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = SimpleVectorStore(collection_name=CHROMA_COLLECTION_NAME)
    indexer = WorkspaceIndexer(str(root), embeddings, vector_store)

    stats = indexer.index()
    print(f"\n✅ Indexing Stats: {stats}")
    print(f"✅ Vector store optimized for Phi3:mini (4K context, code-focused)")
    print(f"✅ Run main.py to start querying with Phi3:mini\n")