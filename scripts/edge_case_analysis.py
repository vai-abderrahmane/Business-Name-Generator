import json, argparse, collections

CATEGORIES = {
    'repetition': lambda n: len(set(n.lower())) < max(3, len(n)//3),
    'trop_long': lambda n: len(n) > 15,
    'trop_court': lambda n: len(n) < 3,
    'caracteres_speciaux': lambda n: any(c for c in n if not (c.isalnum() or c in ['-'])),
}


def classify(name: str):
    matched = [k for k, fn in CATEGORIES.items() if fn(name)]
    return matched or ['ok']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='predictions jsonl with pred_name')
    ap.add_argument('--out', default='outputs/edge_cases.jsonl')
    args = ap.parse_args()

    rows = []
    with open(args.pred, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out_rows = []
    freq = collections.Counter()
    for r in rows:
        cats = classify(r['pred_name'])
        for c in cats:
            freq[c] += 1
        out_rows.append({**r, 'categories': cats})

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('Frequencies:', dict(freq))

if __name__ == '__main__':
    main()
