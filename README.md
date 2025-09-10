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
# Business Name Generator

An AI-powered business name generation system using TinyLlama-1.1B-Chat with multiple approaches: baseline, enhanced prompt engineering, and LoRA fine-tuning.

##  Features

- **Multiple Generation Approaches**: Baseline, Enhanced, and Fine-tuned models
- **Real-time API**: FastAPI-based REST service
- **Quality Evaluation**: 4-criteria scoring system (Relevance, Originality, Readability, Credibility)  
- **Security Filtering**: Content safety and inappropriate term detection
- **Interactive Testing**: Jupyter notebook with comprehensive analysis
- **Production Ready**: Optimized for deployment with minimal resource usage

##  Project Structure

```
ai/
├── notebooks/
│   └── 01_exploration.ipynb     # Main analysis and testing notebook
├── src/
│   └── api/
│       ├── app.py              # FastAPI application
│       └── requirements.txt    # API dependencies
├── scripts/
│   ├── evaluate_judge.py       # Model evaluation script
│   ├── generate_predictions.py # Batch prediction script
│   └── translation_*.py        # Translation utilities
├── data/
│   └── processed/              # Generated datasets and results
├── requirements.txt            # Main project dependencies
├── start_api.sh               # API startup script (Linux/Mac)
├── start_api.ps1              # API startup script (Windows)
└── README.md                  # This file
```

##  Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/vai-abderrahmane/Business-Name-Generator.git
cd Business-Name-Generator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start API Server
```bash
# Linux/Mac
./start_api.sh

# Windows
./start_api.ps1

# Manual start
cd src/api && uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access API
- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

##  Model Approaches

### 🔹 Approach 1: Baseline Model
- **Description**: TinyLlama-1.1B without fine-tuning
- **Features**: Zero-shot generation, fast inference
- **Use Case**: Immediate deployment, minimal resources

### 🔹 Approach 2: Enhanced Prompt Engineering  
- **Description**: Optimized prompts and generation parameters
- **Features**: Improved quality, contextual understanding
- **Use Case**: Better results with no additional training

### 🔹 Approach 3: LoRA Fine-tuned Model
- **Description**: Low-Rank Adaptation fine-tuned on business names
- **Features**: Domain-specific optimization, 0.4% parameter updates
- **Use Case**: Maximum quality for specialized requirements

##  Usage Examples

### API Request
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"description": "AI-powered fitness coaching platform"}
)

names = response.json()["suggestions"]
print(names)  # ["FitCore", "SmartTech", "ActiveAI"]
```

### Jupyter Notebook
Open `notebooks/01_exploration.ipynb` for:
- Interactive testing with custom inputs
- Model comparison and evaluation
- Performance analysis and visualization
- Deployment readiness assessment

##  Evaluation Metrics

The system uses a 4-criteria scoring system:

| Metric | Description | Range |
|--------|-------------|-------|
| **Relevance** | How well the name matches the business description | 0-5 |
| **Originality** | Creativity and uniqueness of the name | 0-5 |
| **Readability** | Ease of reading and pronunciation | 0-5 |
| **Credibility** | Professional appearance and trustworthiness | 0-5 |

## 🛡️ Security Features

- **Content Filtering**: Blocks inappropriate words and terms
- **Pattern Detection**: Identifies suspicious patterns (emails, phones, URLs)
- **Length Validation**: Enforces reasonable name length limits
- **Input Sanitization**: Cleans and validates user inputs

##  Deployment

### Production Deployment
```bash
# Install production dependencies
pip install -r src/api/requirements.txt

# Start with Gunicorn (recommended)
cd src/api
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

##  Performance

- **Model Size**: 1.1B parameters (TinyLlama)
- **Memory Usage**: ~2GB RAM
- **Inference Speed**: ~1-2 seconds per generation
- **Quality Score**: 3.8-4.2/5.0 average across approaches
- **Production Ready**:  Optimized for deployment

##  Development

### Running Tests
```bash
# Run evaluation on sample data
python scripts/evaluate_judge.py --pred data/processed/predictions_baseline.jsonl

# Generate new predictions
python scripts/generate_predictions.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data data/processed/val.jsonl
```

### Jupyter Analysis
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

##  Language Support

- **Fully English**: All code, documentation, and outputs in English
- **International Ready**: Follows global naming conventions
- **Production Standard**: Industry-compliant terminology

##  Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+
- **FastAPI**: 0.100+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB for model cache

##  Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Abderrahmane** - [vai-abderrahmane](https://github.com/vai-abderrahmane)

##  Acknowledgments

- TinyLlama team for the efficient language model
- Hugging Face for the transformers library
- FastAPI for the excellent web framework

---

