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
            questions.append(json.loads(line.strip()))
    return questions

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
            predictions.append({
                "id": q['id'],
                "question": q['question'],
                "answer": result['answer'],
                "arxiv_ids": result['arxiv_ids'],
                "config": "baseline"
            })
            time.sleep(2)
        except Exception as e:
            print(f"Failed: {e}")
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
            predictions.append({
                "id": q['id'],
                "question": q['question'],
                "answer": result['answer'],
                "arxiv_ids": result['arxiv_ids'],
                "config": "full_agent",
                "rounds": result['rounds']
            })
            time.sleep(2)
        except Exception as e:
            print(f"Failed: {e}")
    save_predictions(predictions, "full_agent")
    return predictions

def run_ablation_no_planner(questions):
    print("\n" + "="*50)
    print("RUNNING ABLATION - NO PLANNER")
    print("="*50)
    from agent import retriever, reflector, synthesizer, citation_verifier, MAX_ROUNDS
    predictions = []
    for q in questions:
        print(f"\nQuestion {q['id']}: {q['question'][:60]}...")
        try:
            # skip planner — search original question directly
            all_chunks = []
            all_metadata = []
            for round_num in range(1, MAX_ROUNDS + 1):
                chunks, metas = retriever([q['question']])
                all_chunks.extend(chunks)
                all_metadata.extend(metas)
                if reflector(q['question'], all_chunks, round_num):
                    break
            answer, arxiv_ids = synthesizer(q['question'], all_chunks, all_metadata)
            predictions.append({
                "id": q['id'],
                "question": q['question'],
                "answer": answer,
                "arxiv_ids": arxiv_ids,
                "config": "no_planner"
            })
            time.sleep(2)
        except Exception as e:
            print(f"Failed: {e}")
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
            # one round only — no reflection
            sub_questions = planner(q['question'])
            chunks, metas = retriever(sub_questions)
            answer, arxiv_ids = synthesizer(q['question'], chunks, metas)
            predictions.append({
                "id": q['id'],
                "question": q['question'],
                "answer": answer,
                "arxiv_ids": arxiv_ids,
                "config": "no_reflector"
            })
            time.sleep(2)
        except Exception as e:
            print(f"Failed: {e}")
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
            # no citation verification step
            predictions.append({
                "id": q['id'],
                "question": q['question'],
                "answer": answer,
                "arxiv_ids": arxiv_ids,
                "config": "no_citation_verifier"
            })
            time.sleep(2)
        except Exception as e:
            print(f"Failed: {e}")
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