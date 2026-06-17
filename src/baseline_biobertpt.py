#!/usr/bin/env python3
"""
Baseline BioBERTpt (3 classes) para extracao de relacao -- versao minima.

Objetivo: medir quao bem o BioBERTpt (encoder clinico em portugues) resolve a
tarefa de 3 classes {negation_of, associated_with, no_relation}, com foco no
F1 da classe `negation_of`. E a regua para um modelo focado em negacao depois.

Representacao da entrada: para cada par-candidato (e1, e2) inserimos marcadores
de entidade no texto -- [E1]...[/E1] e [E2]...[/E2] (Soares et al., 2019,
"Matching the Blanks"). Os marcadores sao tokens especiais. Recortamos uma
janela de +-ctx_chars ao redor do par para que as duas entidades sempre caibam
no limite de tokens.

Desbalanceamento: `no_relation` domina (~99%). Usamos CrossEntropy com
class_weight=balanced (peso inversamente proporcional a frequencia).

Selecao de modelo: treina no train, escolhe a melhor epoca pelo macro-F1 no dev,
reporta UMA vez no test. O test nunca escolhe nada.

Metrica: macro-F1, micro-F1 e F1 por classe (sklearn). Sem bootstrap/IC por
opcao de escopo -- o numero que interessa para "detecta bem negacao?" e o F1
da classe `negation_of`.

Uso (CPU funciona, mas e lento; ideal GPU/Colab):
    python src/baseline_biobertpt.py --splits-dir data/splits \
        --model pucpr/biobertpt-all --epochs 3 --batch-size 16 \
        --out results/baseline_biobertpt.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import iter_candidate_pairs  # noqa: E402

LABELS = ["negation_of", "associated_with", "no_relation"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE = "[E1]", "[/E1]", "[E2]", "[/E2]"
MARKER_TOKENS = [E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_marked_window(text, e1, e2, ctx_chars):
    """Recorta janela centrada no par e insere os marcadores de entidade."""
    lo, hi = min(e1["start"], e2["start"]), max(e1["end"], e2["end"])
    ws, we = max(0, lo - ctx_chars), min(len(text), hi + ctx_chars)
    window = text[ws:we]
    inserts: dict[int, list[tuple[int, str]]] = {}

    def add(i, marker, prio):
        inserts.setdefault(i, []).append((prio, marker))

    add(e1["start"] - ws, E1_OPEN + " ", 1)
    add(e1["end"] - ws, " " + E1_CLOSE, 0)
    add(e2["start"] - ws, E2_OPEN + " ", 1)
    add(e2["end"] - ws, " " + E2_CLOSE, 0)

    out = []
    for i in range(len(window) + 1):
        if i in inserts:
            for _, m in sorted(inserts[i]):
                out.append(m)
        if i < len(window):
            out.append(window[i])
    return "".join(out)


def build_dataset(docs, max_gap, ctx_chars):
    texts, labels = [], []
    for doc in docs:
        for c in iter_candidate_pairs(doc, max_gap=max_gap):
            texts.append(build_marked_window(doc["text"], c["e1"], c["e2"], ctx_chars))
            labels.append(LABEL2ID[c["label"]])
    return texts, labels


def set_all_seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(tokenizer, texts, labels, max_length, batch_size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return texts[i], labels[i]

    def collate(batch):
        bt, bl = zip(*batch)
        enc = tokenizer(list(bt), truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(bl)

    g = torch.Generator().manual_seed(seed)
    return DataLoader(DS(), batch_size=batch_size, shuffle=shuffle,
                      generator=g, collate_fn=collate)


def predict(model, loader, device):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, attn, _ in loader:
            logits = model(input_ids=ids.to(device),
                           attention_mask=attn.to(device)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default="results/baseline_biobertpt.json")
    ap.add_argument("--model", default="pucpr/biobertpt-all")
    ap.add_argument("--max-gap", type=int, default=75)
    ap.add_argument("--ctx-chars", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import classification_report, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = Path(args.splits_dir)
    Xtr, ytr = build_dataset(list(read_jsonl(sp / "train.jsonl")), args.max_gap, args.ctx_chars)
    Xdv, ydv = build_dataset(list(read_jsonl(sp / "dev.jsonl")), args.max_gap, args.ctx_chars)
    Xte, yte = build_dataset(list(read_jsonl(sp / "test.jsonl")), args.max_gap, args.ctx_chars)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    tr_loader = make_loader(tok, Xtr, ytr, args.max_length, args.batch_size, True, args.seed)
    dv_loader = make_loader(tok, Xdv, ydv, args.max_length, args.batch_size, False, args.seed)
    te_loader = make_loader(tok, Xte, yte, args.max_length, args.batch_size, False, args.seed)

    if args.class_weight == "balanced":
        counts = np.bincount(ytr, minlength=len(LABELS)).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (len(LABELS) * counts)
        weights = torch.tensor(w, dtype=torch.float).to(device)
    else:
        weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(args.warmup_ratio * total), total)

    ydv_str = [ID2LABEL[i] for i in ydv]
    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for ids, attn, lab in tr_loader:
            opt.zero_grad()
            logits = model(input_ids=ids.to(device),
                           attention_mask=attn.to(device)).logits
            loss = loss_fn(logits, lab.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        dev_pred = [ID2LABEL[i] for i in predict(model, dv_loader, device)]
        macro = f1_score(ydv_str, dev_pred, labels=LABELS, average="macro", zero_division=0)
        history.append({"epoch": ep, "train_loss": running / max(1, len(tr_loader)),
                        "dev_macro_f1": macro})
        print(f"[epoca {ep}] loss={running/max(1,len(tr_loader)):.4f} dev_macro_f1={macro:.4f}")
        if macro > best_f1:
            best_f1 = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    y_pred = [ID2LABEL[i] for i in predict(model, te_loader, device)]
    y_true = [ID2LABEL[i] for i in yte]

    macro = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class = {l: float(per_class[i]) for i, l in enumerate(LABELS)}
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)

    result = {
        "model": args.model, "seed": args.seed, "device": str(device),
        "config": {"max_gap": args.max_gap, "ctx_chars": args.ctx_chars,
                   "max_length": args.max_length, "epochs": args.epochs,
                   "batch_size": args.batch_size, "lr": args.lr,
                   "class_weight": args.class_weight},
        "n_candidates": {"train": len(ytr), "dev": len(ydv), "test": len(yte)},
        "dev_history": history,
        "test_macro_f1": macro, "test_micro_f1": micro,
        "test_f1_per_class": per_class,
        "sklearn_report": report,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                         sort_keys=True), encoding="utf-8")

    print("=" * 60)
    print("BASELINE BioBERTpt (3 classes)")
    print("=" * 60)
    print(f"  candidatos: train={len(ytr)} dev={len(ydv)} test={len(yte)}")
    print(f"  Macro-F1: {macro:.4f}   Micro-F1: {micro:.4f}")
    print(f"  >> negation_of F1: {per_class['negation_of']:.4f} <<")
    print("  F1 por classe:")
    for l in LABELS:
        print(f"    {l:<18s} {per_class[l]:.4f}")
    print(f"  resultado -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
