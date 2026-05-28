import arxiv
import os
import time

PAPERS_DIR = "papers"
os.makedirs(PAPERS_DIR, exist_ok=True)

SEARCH_QUERIES = [
    "LLM agents",
    "agentic RAG",
    "tool use language model",
    "agent memory",
    "agent benchmarks",
    "computer use agents",
]

def scrape_papers(max_papers=600):
    client = arxiv.Client()
    downloaded = 0

    for query in SEARCH_QUERIES:
        if downloaded >= max_papers:
            break

        print(f"Searching for: {query}")
        time.sleep(3)

        search = arxiv.Search(
            query=query,
            max_results=50,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = list(client.results(search))
        print(f"Got {len(results)} results")
        
        for paper in results:
            if downloaded >= max_papers:
                break

            print(f"Found: {paper.title[:50]} | {paper.published.date()}")

            filename = f"{paper.get_short_id().replace('/', '_')}.pdf"
            filepath = os.path.join(PAPERS_DIR, filename)

            if not os.path.exists(filepath):
                try:
                  import requests
                  pdf_url = paper.pdf_url
                  response = requests.get(pdf_url)
                  with open(filepath, 'wb') as f:
                   f.write(response.content)
                  print(f"✓ Downloaded: {paper.title[:50]}")
                  downloaded += 1
                  time.sleep(2)
                except Exception as e:
                  print(f"✗ Failed: {e}")

    print(f"\nTotal papers downloaded: {downloaded}")

if __name__ == "__main__":
    scrape_papers()