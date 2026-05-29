import json
import os
import time
from baseline import answer_question as baseline_answer
from agent import run_agent

QUESTIONS_FILE = "eval/questions.jsonl"
PREDICTIONS_DIR = "predictions"

def load_questions():
    questions = []
    with open(QUESTIONS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def clean_arxiv_id(arxiv_id):
    # remove version suffix e.g. 2605.28773v1 -> 2605.28773
    return arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

def format_prediction(q_id, answer, arxiv_ids):
    cited_papers = list(set([clean_arxiv_id(aid) for aid in arxiv_ids]))
    return {
        "id": q_id,
        "answer": answer,
        "cited_papers": cited_papers
    }

def save_predictions(predictions, config_name):
    filepath = os.path.join(PREDICTIONS_DIR, f"{config_name}.jsonl")
    with open(filepath, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Saved {len(predictions)} predictions to {filepath}")

def run_baseline(questions):
    print("\n" + "="*50)
    print("RUNNING BASELINE")
    print("="*50)
    predictions = []
    for q in questions:
        print(f"\nQuestion {q['id']}: {q['question'][:60]}...")
        try:
            result = baseline_answer(q['question'])
            pred = format_prediction(q['id'], result['answer'], result['arxiv_ids'])
            predictions.append(pred)
            time.sleep(3)
        except Exception as e:
            print(f"Failed: {e}")
            predictions.append({"id": q['id'], "answer": "Unable to answer.", "cited_papers": []})
    save_predictions(predictions, "baseline")
    return predictions

def run_full_agent(questions):
    print("\n" + "="*50)
    print("RUNNING FULL AGENT")
    print("="*50)
    predictions = []
    for q in questions:
        print(f"\nQuestion {q['id']}: {q['question'][:60]}...")
        try:
            result = run_agent(q['question'])
            pred = format_prediction(q['id'], result['answer'], result['arxiv_ids'])
            predictions.append(pred)
            time.sleep(3)
        except Exception as e:
            print(f"Failed: {e}")
            predictions.append({"id": q['id'], "answer": "Unable to answer.", "cited_papers": []})
    save_predictions(predictions, "full_agent")
    return predictions

def run_ablation_no_planner(questions):
    print("\n" + "="*50)
    print("RUNNING ABLATION - NO PLANNER")
    print("="*50)
    from agent import retriever, reflector, synthesizer, MAX_ROUNDS
    predictions = []
    for q in questions:
        print(f"\nQuestion {q['id']}: {q['question'][:60]}...")
        try:
            all_chunks = []
            all_metadata = []
            for round_num in range(1, MAX_ROUNDS + 1):
                chunks, metas = retriever([q['question']])
                all_chunks.extend(chunks)
                all_metadata.extend(metas)
                if reflector(q['question'], all_chunks, round_num):
                    break
            answer, arxiv_ids = synthesizer(q['question'], all_chunks, all_metadata)
            pred = format_prediction(q['id'], answer, arxiv_ids)
            predictions.append(pred)
            time.sleep(3)
        except Exception as e:
            print(f"Failed: {e}")
            predictions.append({"id": q['id'], "answer": "Unable to answer.", "cited_papers": []})
    save_predictions(predictions, "no_planner")
    return predictions

def run_ablation_no_reflector(questions):
    print("\n" + "="*50)
    print("RUNNING ABLATION - NO REFLECTOR")
    print("="*50)
    from agent import planner, retriever, synthesizer
    predictions = []
    for q in questions:
        print(f"\nQuestion {q['id']}: {q['question'][:60]}...")
        try:
            sub_questions = planner(q['question'])
            chunks, metas = retriever(sub_questions)
            answer, arxiv_ids = synthesizer(q['question'], chunks, metas)
            pred = format_prediction(q['id'], answer, arxiv_ids)
            predictions.append(pred)
            time.sleep(3)
        except Exception as e:
            print(f"Failed: {e}")
            predictions.append({"id": q['id'], "answer": "Unable to answer.", "cited_papers": []})
    save_predictions(predictions, "no_reflector")
    return predictions

def run_ablation_no_citation_verifier(questions):
    print("\n" + "="*50)
    print("RUNNING ABLATION - NO CITATION VERIFIER")
    print("="*50)
    from agent import planner, retriever, reflector, synthesizer, MAX_ROUNDS
    predictions = []
    for q in questions:
        print(f"\nQuestion {q['id']}: {q['question'][:60]}...")
        try:
            sub_questions = planner(q['question'])
            all_chunks = []
            all_metadata = []
            for round_num in range(1, MAX_ROUNDS + 1):
                chunks, metas = retriever(sub_questions)
                all_chunks.extend(chunks)
                all_metadata.extend(metas)
                if reflector(q['question'], all_chunks, round_num):
                    break
            answer, arxiv_ids = synthesizer(q['question'], all_chunks, all_metadata)
            pred = format_prediction(q['id'], answer, arxiv_ids)
            predictions.append(pred)
            time.sleep(3)
        except Exception as e:
            print(f"Failed: {e}")
            predictions.append({"id": q['id'], "answer": "Unable to answer.", "cited_papers": []})
    save_predictions(predictions, "no_citation_verifier")
    return predictions

if __name__ == "__main__":
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    run_baseline(questions)
    run_full_agent(questions)
    run_ablation_no_planner(questions)
    run_ablation_no_reflector(questions)
    run_ablation_no_citation_verifier(questions)

    print("\n✓ All configurations complete!")
    print("Check predictions/ folder for results")