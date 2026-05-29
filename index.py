import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PAPERS_DIR = "papers"
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # overlapping words between chunks

# initialize embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("papers")

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def index_papers():
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} papers to index")

    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        arxiv_id = pdf_file.replace('.pdf', '').replace('_', '/')

        # skip if already indexed
        existing = collection.get(where={"arxiv_id": arxiv_id})
        if existing['ids']:
            continue

        # extract and chunk text
        text = extract_text(pdf_path)
        if not text.strip():
            continue

        chunks = chunk_text(text)

        # embed and store
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                ids=[f"{arxiv_id}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"arxiv_id": arxiv_id, "chunk_id": i}]
            )

    print(f"Indexing complete! Total chunks: {collection.count()}")

if __name__ == "__main__":
    index_papers()