import argparse, json, random, os, datetime, hashlib
from pathlib import Path

SECTORS = [
    "café de quartier", "startup IA", "agence marketing", "association sportive", "ONG climat",
    "studio de design", "fintech", "cabinet juridique", "e-commerce mode", "foodtech durable",
    "coworking", "plateforme éducative", "cybersecurité", "agritech", "biotech", "mobilité verte",
    "tourisme local", "media en ligne", "formation pro", "blockchain conformité"
]

TONE = ["moderne", "classique", "tech", "élégant", "convivial", "premium", "accessible", "innovant"]

DOMAINS_SUFFIX = [".com", ".io", ".ai", ".fr", ".co", ".net"]

ADJECTIVES = ["nova", "prime", "hyper", "blue", "green", "clear", "swift", "alpha", "neo", "smart", "terra", "stellar", "nexus", "lumina", "core", "pulse"]
NOUNS = ["labs", "hub", "cloud", "studio", "works", "data", "tech", "system", "gate", "flow", "mind", "forge", "loop", "grid", "stack"]

BLOCKLIST = {"shit", "fuck", "bitch", "con"}  # simple safety


def slugify(name: str) -> str:
    s = name.lower().replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a")
    return ''.join(c for c in s if c.isalnum() or c in ['-'])


def gen_name(seed_tuple):
    adj1, noun1 = seed_tuple
    base = f"{adj1}{noun1}".lower()
    variants = [base, base + "ly", base + "ia", base + "ity", base.replace("a", "o", 1)]
    return random.choice(variants).capitalize()


def generate_record(i: int):
    sector = random.choice(SECTORS)
    tone = random.choice(TONE)
    size = random.choice(["TPE", "PME", "scale-up", "indépendant"])
    locale = random.choice(["France", "Europe", "Global", "Local"])
    desc = f"Une {sector} {tone} ({size}) opérant en {locale}."  # simplistic
    # candidate company name
    seed = (random.choice(ADJECTIVES), random.choice(NOUNS))
    company_name = gen_name(seed)
    # domain suggestion
    domain_base = slugify(company_name)
    domain = domain_base + random.choice(DOMAINS_SUFFIX)
    # ensure no blocklist tokens
    if any(b in company_name.lower() for b in BLOCKLIST):
        company_name = company_name[0] + hashlib.md5(company_name.encode()).hexdigest()[:5]
    return {
        "id": i,
        "description": desc,
        "sector": sector,
        "tone": tone,
        "size": size,
        "locale": locale,
        "target_name": company_name,
        "target_domain": domain,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--out', type=str, default='data/processed/dataset.jsonl')
    args = ap.parse_args()

    random.seed(42)
    records = [generate_record(i) for i in range(args.n)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    meta = {
        "generated_at": datetime.datetime.utcnow().isoformat() + 'Z',
        "num_records": len(records),
        "method": "synthetic combinatorial generation",
        "fields": list(records[0].keys()) if records else [],
    }
    with open('data/processed/dataset_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} records to {out_path}")

if __name__ == '__main__':
    main()
