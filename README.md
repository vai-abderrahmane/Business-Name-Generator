# Projet: Générateur de Noms d'Entreprise Assisté par LLM

Cycle complet: dataset → modèle → évaluation (LLM-as-a-Judge) → edge cases → amélioration → sécurité → rapport.

## Structure
```
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── modeling/
│   ├── eval/
│   ├── safety/
│   └── api/
├── scripts/
├── reports/
└── requirements.txt
```

## Étapes
1. Génération du dataset synthétique.
2. Fine-tuning (LoRA) d'un modèle open-source.
3. Évaluation automatique (heuristique locale) des propositions.
4. Analyse d'edge cases (optionnelle).
5. Garde-fous de sécurité (regex / pipeline).
6. API FastAPI de génération.

## Environnement
Créer un environnement virtuel puis installer:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Tracking
Utiliser Weights & Biases (optionnel):
```
wandb login
WANDB_PROJECT=company-name-gen
```

## Lancement rapide
```
# 1. Dataset synthétique
python scripts/generate_synthetic_dataset.py --n 500

# 2. Fine-tuning LoRA (exemple minimal – utiliser aussi le notebook si besoin)
python scripts/finetune_with_metrics.py \
	--model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
	--train_file data/processed/train.jsonl \
	--val_file data/processed/val.jsonl \
	--output_dir outputs/lora_run \
	--num_epochs 1 --batch_size 2

# 3. Génération de prédictions avec le modèle (adapter --model si LoRA sauvegardé)
python scripts/generate_predictions.py --model outputs/lora_run --data data/processed/dataset.jsonl --out outputs/predictions.jsonl

# 4. Évaluation heuristique locale
python scripts/evaluate_judge.py --pred outputs/predictions.jsonl --out outputs/eval_scores.jsonl

# 5. (Option) Edge cases
python scripts/edge_case_analysis.py --pred outputs/predictions.jsonl --out outputs/edge_cases.jsonl

# 6. Lancer l'API (PowerShell)
./start_api.ps1
```

## Scripts conservés (essentiels)
- `scripts/generate_synthetic_dataset.py` : génération jeu synthétique
- `scripts/finetune_with_metrics.py` : fine-tuning LoRA + métriques
- `scripts/generate_predictions.py` : inférence lot
- `scripts/evaluate_judge.py` : scoring heuristique
- `scripts/edge_case_analysis.py` (optionnel)
- `start_api.ps1` / `start_api.sh` : démarrage API

Scripts retirés : variantes LLM judge (OpenAI / Anthropic), doublon LoRA simple, filtre sécurité redondant, script API doublon. Ces fonctionnalités sont couvertes par les versions unifiées restantes.

## Licence
Usage expérimental.
# Business-Name-Generator
