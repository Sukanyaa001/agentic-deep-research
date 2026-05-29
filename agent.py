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

MAX_ROUNDS = 5  # max reflection rounds

def llm_call(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

def retrieve(query, n_results=5):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results['documents'][0], results['metadatas'][0]

# ── PLANNER ──
def planner(question):
    prompt = f"""Break this research question into 2-4 specific sub-questions that together 
will fully answer it. Return ONLY a numbered list, nothing else.

Question: {question}"""
    response = llm_call(prompt)
    sub_questions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # remove numbering
            sub_q = line.split('.', 1)[-1].strip()
            sub_questions.append(sub_q)
    return sub_questions

# ── RETRIEVER ──
def retriever(sub_questions):
    all_chunks = []
    all_metadata = []
    for q in sub_questions:
        chunks, metas = retrieve(q)
        all_chunks.extend(chunks)
        all_metadata.extend(metas)
    return all_chunks, all_metadata

# ── REFLECTOR ──
def reflector(question, chunks, round_num):
    context_preview = "\n".join(chunks[:3])[:1000]
    prompt = f"""You are evaluating if we have enough information to answer a research question.

Question: {question}
Current round: {round_num}/{MAX_ROUNDS}
Number of chunks retrieved: {len(chunks)}
Sample context: {context_preview}

Do we have enough information to write a comprehensive answer? 
Reply with only YES or NO."""
    response = llm_call(prompt)
    return "YES" in response.upper()

# ── SYNTHESIZER ──
def synthesizer(question, chunks, metadatas):
    context = ""
    arxiv_ids = []
    for chunk, meta in zip(chunks, metadatas):
        arxiv_id = meta['arxiv_id']
        context += f"[{arxiv_id}]: {chunk}\n\n"
        if arxiv_id not in arxiv_ids:
            arxiv_ids.append(arxiv_id)

    prompt = f"""You are a research assistant. Write a comprehensive answer using ONLY 
the provided context. Include inline citations using arXiv IDs like [arxiv_id].

Context:
{context[:2000]}

Question: {question}

Answer:"""
    answer = llm_call(prompt)
    return answer, arxiv_ids

# ── CITATION VERIFIER ──
def citation_verifier(answer, chunks, metadatas):
    prompt = f"""Check if the citations in this answer are supported by the provided context.
List any citations that seem unsupported.

Answer: {answer[:1000]}

Context summary: {len(chunks)} chunks from {len(set(m['arxiv_id'] for m in metadatas))} papers.

Are all citations supported? Reply YES or NO and briefly explain."""
    return llm_call(prompt)

# ── MAIN AGENT ──
def run_agent(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print('='*50)

    # step 1 — plan
    print("\n[PLANNER] Breaking question into sub-questions...")
    sub_questions = planner(question)
    for i, q in enumerate(sub_questions):
        print(f"  {i+1}. {q}")

    # step 2 — retrieve + reflect loop
    all_chunks = []
    all_metadata = []
    
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n[RETRIEVER] Round {round_num} - searching...")
        new_chunks, new_metas = retriever(sub_questions)
        all_chunks.extend(new_chunks)
        all_metadata.extend(new_metas)
        print(f"  Total chunks so far: {len(all_chunks)}")

        print(f"[REFLECTOR] Checking if enough information...")
        has_enough = reflector(question, all_chunks, round_num)
        
        if has_enough:
            print(f"  ✓ Enough information found!")
            break
        else:
            print(f"  ✗ Need more information, searching again...")
            # refine sub-questions for next round
            sub_questions = [q + " detailed analysis" for q in sub_questions]

    # step 3 — synthesize
    print("\n[SYNTHESIZER] Writing answer...")
    answer, arxiv_ids = synthesizer(question, all_chunks, all_metadata)

    # step 4 — verify citations
    print("\n[CITATION VERIFIER] Checking citations...")
    verification = citation_verifier(answer, all_chunks, all_metadata)
    print(f"  {verification[:100]}")

    return {
        "question": question,
        "answer": answer,
        "arxiv_ids": arxiv_ids,
        "config": "full_agent",
        "rounds": round_num
    }

if __name__ == "__main__":
    question = "How do LLM agents handle memory and what are the different approaches?"
    result = run_agent(question)
    print(f"\n{'='*50}")
    print(f"FINAL ANSWER:\n{result['answer']}")
    print(f"\nCited papers: {result['arxiv_ids']}")
    print(f"Rounds used: {result['rounds']}")