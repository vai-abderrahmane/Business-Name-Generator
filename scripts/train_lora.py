import argparse, json, os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

# Minimal LoRA fine-tune script (causal LM style) expecting jsonl with fields: description,target_name

def load_jsonl(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            lines.append(json.loads(l))
    return lines


def build_prompt(example):
    return f"### Description:\n{example['description']}\n### Task:\nPropose a nom d'entreprise adapté.\n### Réponse:\n{example['target_name']}"  # supervised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    # Legacy param --data, plus alias --train_file/--val_file pour compat notebook
    ap.add_argument('--data', type=str, required=False, help='Fichier d\'entrainement (legacy)')
    ap.add_argument('--train_file', type=str, help='Alias de --data')
    ap.add_argument('--val_file', type=str, help='(Optionnel, ignoré pour l\'instant)')
    ap.add_argument('--output', type=str, default='checkpoints/lora')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gradient_checkpointing', action='store_true', help='Activer gradient checkpointing pour réduire la RAM GPU')
    ap.add_argument('--quant', type=str, default='none', choices=['none','8bit','4bit'], help='Chargement quantifié (bitsandbytes requis)')
    args = ap.parse_args()

    # Résolution du chemin d'entrainement
    train_path = args.data or args.train_file
    if not train_path:
        raise ValueError('Aucun fichier train fourni (--data ou --train_file).')
    train_path = Path(train_path)
    if not train_path.exists():
        # tentative: relatif à cwd parent si execution depuis scripts/
        alt = Path.cwd().parent / train_path
        if alt.exists():
            train_path = alt
        else:
            # tentative: remonter 2 niveaux
            alt2 = Path.cwd().parent.parent / train_path
            if alt2.exists():
                train_path = alt2
    if not train_path.exists():
        raise FileNotFoundError(f"Fichier train introuvable après résolutions: {train_path}")
    print(f"[INFO] Fichier train utilisé: {train_path}")

    records = load_jsonl(str(train_path))

    # Build dataset
    texts = [build_prompt(r) for r in records]
    ds_dict = {"text": texts}

    from datasets import Dataset
    dataset = Dataset.from_dict(ds_dict)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=args.max_len)

    tokenized = dataset.map(tok_fn, batched=True, remove_columns=['text'])

    load_kwargs = {}
    if args.quant in ['8bit','4bit']:
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs['device_map'] = 'auto'
            if args.quant == '8bit':
                load_kwargs['load_in_8bit'] = True
            else:
                load_kwargs['load_in_4bit'] = True
        except ImportError:
            print('[WARN] bitsandbytes introuvable – fallback sans quantization')
    else:
        # device_map auto si GPU dispo
        load_kwargs['device_map'] = 'auto'

    if args.fp16:
        load_kwargs['torch_dtype'] = torch.float16

    print(f"[INFO] Chargement modèle avec kwargs: {load_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print('[INFO] Gradient checkpointing activé')

    # Détection dynamique des modules cibles LoRA pour compatibilité multi-architectures
    candidate_substrings = [
        'q_proj','k_proj','v_proj','o_proj',          # Architectures type LLaMA / Mistral
        'gate_proj','up_proj','down_proj',            # Autres variantes
        'c_attn','c_proj'                            # GPT2 style
    ]
    present = set()
    for name, module in model.named_modules():
        for cand in candidate_substrings:
            if cand in name:
                present.add(cand)
    target_modules = sorted(present) if present else ['q_proj','v_proj']
    print(f"[LoRA] Target modules: {target_modules}")

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')

    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy='epoch',
        fp16=args.fp16,
        gradient_accumulation_steps=4,
        optim='adamw_torch',
        warmup_steps=10,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"Saved LoRA adapter + tokenizer to {args.output}")

if __name__ == '__main__':
    main()
