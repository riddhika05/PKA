import os
from pathlib import Path
from typing import Dict, Any
import chromadb

from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    SentenceSplitter,
    CodeSplitter,
    MarkdownNodeParser,
    JSONNodeParser
)
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.readers.base import BaseReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.readers.file import (
    FlatReader,
    CSVReader,
    MarkdownReader,
    PDFReader,
    PptxReader,
    DocxReader,
    ImageReader
)

WORKSPACE_NAME = "MyRAG_Knowledge_Base"
WORKSPACE_DRIVE = "C:"
WORKSPACE_PATH = os.path.join(WORKSPACE_DRIVE, os.sep, WORKSPACE_NAME)

CHROMA_COLLECTION_NAME = "documents"
CHROMA_STORAGE_PATH = "rag_chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP = 200

CODE_LANGUAGE = "python"
CODE_CHUNK_SIZE = 1000
CODE_CHUNK_OVERLAP = 200
DOC_CHUNK_SIZE = 512
DOC_CHUNK_OVERLAP = 50

BLACKLIST_FOLDERS = {
    "node_modules", "__pycache__", "venv", "env", ".git", ".idea", ".vscode",
    "dist", "build", "target", ".next", ".nuxt",
    "bin", "obj", "packages", "Migrations"
}

def get_default_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=DOC_CHUNK_SIZE,
        chunk_overlap=DOC_CHUNK_OVERLAP,
    )

def get_code_splitter(language: str = CODE_LANGUAGE) -> CodeSplitter:
    return CodeSplitter(
        language=language,
        chunk_lines=40,
        chunk_lines_overlap=15,
        max_chars=CODE_CHUNK_SIZE,
    )

def get_json_parser() -> JSONNodeParser:
    return JSONNodeParser()

def get_markdown_parser() -> MarkdownNodeParser:
    return MarkdownNodeParser()

FILE_EXTRACTOR: Dict[str, BaseReader] = {
    ".py": FlatReader(),
    ".js": FlatReader(),
    ".ts": FlatReader(),
    ".java": FlatReader(),
    ".cs": FlatReader(),
    ".cshtml": FlatReader(),
    ".cpp": FlatReader(),
    ".c": FlatReader(),
    ".jsx": FlatReader(),
    ".tsx": FlatReader(),
    ".html": FlatReader(),
    ".css": FlatReader(),
    ".sh": FlatReader(),
    ".ps1": FlatReader(),
    ".rb": FlatReader(),
    ".php": FlatReader(),
    ".swift": FlatReader(),
    ".kt": FlatReader(),
    ".go": FlatReader(),
    ".rs": FlatReader(),
    ".yaml": FlatReader(),
    ".yml": FlatReader(),
    ".json": FlatReader(),
    ".xml": FlatReader(),
    ".pdf": PDFReader(),
    ".docx": DocxReader(),
    ".pptx": PptxReader(),
    ".csv": CSVReader(),
    ".txt": FlatReader(),
    ".md": MarkdownReader(),
    ".log": FlatReader(),
    ".jpg": ImageReader(),
    ".jpeg": ImageReader(),
    ".png": ImageReader(),
}

FILE_SPLITTER_FACTORY: Dict[str, Any] = {
    ".py": lambda: get_code_splitter("python"),
    ".js": lambda: get_code_splitter("javascript"),
    ".ts": lambda: get_code_splitter("typescript"),
    ".java": lambda: get_code_splitter("java"),
    ".cs": lambda: get_default_splitter,
    ".cshtml": lambda: get_default_splitter,
    ".cpp": lambda: get_code_splitter("cpp"),
    ".c": lambda: get_code_splitter("c"),
    ".jsx": lambda: get_code_splitter("javascript"),
    ".tsx": lambda: get_code_splitter("typescript"),
    ".html": lambda: get_code_splitter("html"),
    ".css": get_default_splitter,
    ".sh": get_default_splitter,
    ".rb": lambda: get_code_splitter("ruby"),
    ".go": lambda: get_code_splitter("go"),
    ".rs": lambda: get_code_splitter("rust"),
    ".php": lambda: get_code_splitter("php"),
    ".swift": get_default_splitter,
    ".kt": get_default_splitter,
    ".md": get_markdown_parser,
    ".json": get_json_parser,
    ".yaml": get_default_splitter,
    ".yml": get_default_splitter,
    ".xml": get_default_splitter,
    ".pdf": get_default_splitter,
    ".docx": get_default_splitter,
    ".pptx": get_default_splitter,
    ".csv": get_default_splitter,
    ".txt": get_default_splitter,
    ".log": get_default_splitter,
    ".jpg": get_default_splitter,
    ".jpeg": get_default_splitter,
    ".png": get_default_splitter,
}

_RAG_COMPONENTS = None

def get_rag_components() -> tuple[VectorStoreIndex, Any, HuggingFaceEmbedding]:
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE_CHARS
    Settings.chunk_overlap = CHUNK_OVERLAP
    
    client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
    chroma_collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
    except Exception:
        index = VectorStoreIndex([], storage_context=storage_context)

    return index, chroma_collection, Settings.embed_model


def get_or_create_rag_components():
    global _RAG_COMPONENTS
    if _RAG_COMPONENTS is None:
        _RAG_COMPONENTS = get_rag_components()
    return _RAG_COMPONENTS


def get_file_reader(file_path: Path) -> BaseReader:
    ext = file_path.suffix.lower()
    return FILE_EXTRACTOR.get(ext, FlatReader())


def get_file_splitter(file_path: Path):
    ext = file_path.suffix.lower()
    splitter_or_func = FILE_SPLITTER_FACTORY.get(ext)
    
    if callable(splitter_or_func):
        return splitter_or_func()
    
    return splitter_or_func if splitter_or_func is not None else get_default_splitter()


def get_project_name(file_path: Path, workspace_root: Path) -> str:
    try:
        rel = file_path.relative_to(workspace_root)
        parts = rel.parts
        if len(parts) > 1:
            return parts[0]
        return "root"
    except Exception:
        return "unknown"


def create_ingestion_pipeline(chroma_collection: Any) -> IngestionPipeline:
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    pipeline = IngestionPipeline(
        transformations=[
            get_default_splitter(),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )
    return pipeline