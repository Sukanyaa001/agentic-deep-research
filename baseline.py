import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# initialize
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("papers")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(query, n_results=5):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    return chunks, metadatas

def answer_question(question):
    # single retrieval
    chunks, metadatas = retrieve(question)
    
    # build context
    context = ""
    arxiv_ids = []
    for chunk, meta in zip(chunks, metadatas):
        arxiv_id = meta['arxiv_id']
        context += f"[{arxiv_id}]: {chunk}\n\n"
        if arxiv_id not in arxiv_ids:
            arxiv_ids.append(arxiv_id)
    
    # call LLM
    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
Include inline citations using arXiv IDs like [arxiv_id].

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    return {
        "question": question,
        "answer": answer,
        "arxiv_ids": arxiv_ids,
        "config": "baseline"
    }

if __name__ == "__main__":
    # test it
    question = "How do LLM agents handle memory?"
    result = answer_question(question)
    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nCited papers: {result['arxiv_ids']}")