import requests
import os
import time

PAPERS_DIR = "papers"
os.makedirs(PAPERS_DIR, exist_ok=True)

# Specific papers mentioned in the questions
TARGETED_PAPERS = [
    # Mem0
    "2504.19413",
    # tau-bench
    "2406.12045",
    # OSWorld
    "2404.07972",
    # SWE-agent
    "2405.15793",
    # Agent Interoperability Protocols
    "2505.02279",
    # Agentic RAG survey
    "2501.09136",
    # AppWorld
    "2407.18901",
    # UI-TARS
    "2501.12326",
    # UI-TARS-2
    "2504.02479",
    # OpenHands
    "2407.16741",
    # OS-MAP
    "2503.15116",
    # A-MEM
    "2502.12110",
    # Multi-Turn Multi-Agent Orchestration
    "2503.11875",
    # Multi-Agent Collaboration via Evolving Orchestration
    "2503.05164",
    # Can LLM Agents Really Debate
    "2504.08120",
    # Multi-Agent Collaboration Mechanisms survey
    "2501.06322",
    # Deep Research Agents survey
    "2506.18096",
    # Deep Research autonomous agents survey
    "2502.12345",
    # Open and Reproducible Deep Research
    "2502.19522",
    # From Web Search Towards Agentic Deep Research
    "2503.14463",
    # Reflexion
    "2303.11366",
    # ReAct
    "2210.03629",
    # Self-RAG
    "2310.11511",
    # RAGAS
    "2309.15217",
    # SWE-EVO
    "2502.15358",
    # FinMem
    "2311.13743",
]

def download_paper(arxiv_id):
    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    filepath = os.path.join(PAPERS_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Already exists: {arxiv_id}")
        return True
    
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {arxiv_id}")
            return True
        else:
            print(f"✗ Failed {arxiv_id}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error {arxiv_id}: {e}")
        return False

if __name__ == "__main__":
    print(f"Downloading {len(TARGETED_PAPERS)} targeted papers...")
    success = 0
    for arxiv_id in TARGETED_PAPERS:
        if download_paper(arxiv_id):
            success += 1
        time.sleep(3)
    print(f"\nDone! Downloaded {success}/{len(TARGETED_PAPERS)} papers")
