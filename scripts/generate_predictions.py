import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """### Description:
{description}
### Task:
Suggest a suitable business name.
### Response:
"""

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True, help='jsonl dataset with description')
    ap.add_argument('--out', default='outputs/predictions.jsonl')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--max_new_tokens', type=int, default=16)
    args = ap.parse_args()

    records = load_jsonl(args.data)[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in records:
            prompt = PROMPT_TEMPLATE.format(description=r['description'])
            inputs = tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.9, temperature=0.9)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = text.split('### Response:')[-1].strip().split('\n')[0]
            f.write(json.dumps({
                'description': r['description'],
                'pred_name': pred
            }, ensure_ascii=False) + '\n')

    print(f"Wrote predictions to {args.out}")

if __name__ == '__main__':
    main()
