import argparse, json, os, time
from pathlib import Path
from typing import Dict
from anthropic import Anthropic

CRITERIA = ["pertinence","originalite","lisibilite","credibilite"]
SYSTEM = "Tu évalues des noms d'entreprises. Pour la description et un nom proposé, réponds uniquement en JSON: { 'pertinence':1-5,'originalite':1-5,'lisibilite':1-5,'credibilite':1-5,'commentaire':'...'}"

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

PROMPT = (
"Description: {desc}\nNom proposé: {name}\nDonne l'évaluation JSON:" )

def judge(desc: str, name: str) -> Dict:
    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        system=SYSTEM,
        messages=[{"role":"user","content":PROMPT.format(desc=desc, name=name)}]
    )
    content = ''.join(block.text for block in msg.content if hasattr(block,'text'))
    try:
        return json.loads(content)
    except Exception:
        return {c:0 for c in CRITERIA} | {"commentaire":"parse_error","raw":content}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', default='outputs/eval_scores_anthropic.jsonl')
    ap.add_argument('--sleep', type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    with open(args.pred,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        for r in rows:
            scores = judge(r['description'], r['pred_name'])
            f.write(json.dumps({**r, **scores}, ensure_ascii=False)+'\n')
            time.sleep(args.sleep)

if __name__=='__main__':
    main()
