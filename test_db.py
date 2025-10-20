import chromadb

client = chromadb.PersistentClient(path="rag_chroma_db")
try:
    collection = client.get_collection(name="documents")
    count = collection.count()
    print(f"✓ Database exists")
    print(f"✓ Collection 'documents' found")
    print(f"✓ Total chunks in DB: {count}")
    
    if count > 0:
        # Try a test query
        results = collection.query(
            query_embeddings=[collection.get(limit=1)["embeddings"][0]],
            n_results=1
        )
        print(f"✓ Sample query works: {results['documents'][0][0][:100]}")
    else:
        print("✗ Database is EMPTY - no chunks found")
except Exception as e:
    print(f"✗ Error: {e}")