import argparse, json, os, time, statistics
from pathlib import Path
from typing import List, Dict

CRITERIA = [
    ("relevance", "Does the name match well with the business description?"),
    ("originality", "Is the name creative without being absurd?"),
    ("readability", "Is the name easy to read and pronounce?"),
    ("credibility", "Does the name inspire trust/professionalism?"),
]

SYSTEM_PROMPT = (
    "You are an expert business name evaluator. For each proposal, assign a score from 1 to 5 on each criterion. "
    "Respond in compact JSON with keys: relevance, originality, readability, credibility and a brief comment field."
)

# Placeholder judge using a simple heuristic (no external API here). Real implementation would call GPT-4 / Claude.

def heuristic_judge(name: str, description: str) -> Dict:
    base = len(set(name.lower()))
    relevance = 4 if any(tok in description.lower() for tok in name.lower().split()) else 3
    originality = 3 + (1 if base > 5 else 0)
    readability = 4 if len(name) < 12 else 3
    credibility = 4 if name[0].isupper() else 3
    return {
        "relevance": relevance,
        "originality": originality,
        "readability": readability,
        "credibility": credibility,
        "comment": "heuristic"
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='Predictions jsonl (with description,pred_name)')
    ap.add_argument('--out', default='outputs/eval_scores.jsonl')
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.pred, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scored = []
    for r in rows:
        s = heuristic_judge(r['pred_name'], r['description'])
        scored.append({**r, **s})

    with open(args.out, 'w', encoding='utf-8') as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # aggregate
    agg = {k: statistics.mean([s[k] for s in scored]) for k,_ in CRITERIA}
    print("Averages:", agg)

if __name__ == '__main__':
    main()
