#!/usr/bin/env python3
"""
Nucleo compartilhado dos baselines de extracao de relacao (RECLin-PT).

Este modulo concentra TODA a logica de treino e avaliacao usada pelos baselines
de encoder (BioBERTpt clinico e BERTimbau geral). Os scripts de entrada
(`baseline_biobertpt.py`, `baseline_bertimbau.py`) sao apenas finas cascas que
escolhem o modelo e os caminhos de saida e chamam `run(args, log)` daqui.

POR QUE UM NUCLEO UNICO
-----------------------
A pergunta de pesquisa e: "o pre-treinamento clinico importa para extracao de
relacoes em textos medicos em portugues?". A resposta so e valida se a UNICA
diferenca entre os dois experimentos for o checkpoint de pre-treino. Mantendo
representacao de entrada, loss, scheduler, early stopping, salvamento, metricas,
seed e logging num so lugar, a paridade entre BioBERTpt e BERTimbau e garantida
por construcao -- nao por disciplina de copiar-colar.

ITENS MANTIDOS IDENTICOS ENTRE OS BASELINES
-------------------------------------------
- Entity markers / representacao de entrada: [E1] ... [/E1] e [E2] ... [/E2]
  como tokens especiais (Soares et al., 2019, "Matching the Blanks"), com janela
  de +-ctx_chars ao redor do par.
- Loss: CrossEntropy com class_weight=balanced (negativos `no_relation` dominam).
- Scheduler: linear com warmup (get_linear_schedule_with_warmup).
- Early stopping / selecao de modelo: melhor epoca pelo macro-F1 no DEV; o TEST
  e reportado UMA unica vez.
- Salvamento de checkpoints: `best_model/` (pesos do melhor epoch, formato HF) e
  `last_checkpoint/` (estado completo de retomada: modelo + optimizer + scheduler
  + RNG + epoca + historico), ambos gravados de forma ATOMICA.
- Metricas: Micro-F1, Macro-F1, F1 por classe, classification_report (sklearn) e
  matriz de confusao.
- Seed de reprodutibilidade: set_all_seeds (python/numpy/torch/cuda + cudnn
  deterministico).
- Logging estruturado centralizado: tudo passa por src/utils/logger.py ->
  terminal + logs/pipeline.log.

CHECKPOINT / RETOMADA (--ckpt-dir)
----------------------------------
Ao fim de CADA epoca grava `<ckpt-dir>/last_checkpoint/` com o estado completo de
treino, de forma atomica (escreve em .tmp e renomeia). Quando uma epoca bate o
melhor dev macro-F1, grava tambem `<ckpt-dir>/best_model/` (somente pesos, no
formato `save_pretrained`, pronto para inferencia/compartilhamento). Ao iniciar,
se `last_checkpoint/` existir e a config bater, retoma da PROXIMA epoca -- pulando
as ja concluidas. Aponte --ckpt-dir para uma pasta no Google Drive para
sobreviver a quedas do runtime do Colab. Se todas as epocas ja foram feitas, vai
direto para a avaliacao final.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import iter_candidate_pairs  # noqa: E402
from utils.logger import get_logger  # noqa: E402

# Logger padrao do nucleo. Cada entry-point passa o seu (com o nome do baseline)
# para `run`, que reatribui o global abaixo -- assim o campo [MODULO] do log
# reflete qual baseline esta rodando.
log = get_logger("re_core")

# --------------------------------------------------------------------------- #
# Espaco de rotulos e marcadores de entidade                                  #
# --------------------------------------------------------------------------- #
LABELS = ["negation_of", "associated_with", "no_relation"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE = "[E1]", "[/E1]", "[E2]", "[/E2]"
MARKER_TOKENS = [E1_OPEN, E1_CLOSE, E2_OPEN, E2_CLOSE]

# Nomes das pastas de checkpoint (mantidos identicos entre os baselines).
BEST_MODEL_DIR = "best_model"
LAST_CKPT_DIR = "last_checkpoint"
STATE_FILE = "training_state.pt"


# --------------------------------------------------------------------------- #
# Dados / representacao de entrada                                             #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Reprodutibilidade                                                           #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Matriz de confusao (renderizacao em texto para o log)                        #
# --------------------------------------------------------------------------- #
def render_confusion_matrix(cm, labels):
    """Devolve a matriz de confusao como string alinhada (linhas=verdadeiro,
    colunas=predito). Usada no log; o array tambem vai para o JSON de saida."""
    short = [l[:8] for l in labels]
    head = "true\\pred".ljust(12) + "".join(s.rjust(10) for s in short)
    lines = [head]
    for i, l in enumerate(labels):
        row = l[:11].ljust(12) + "".join(str(int(cm[i][j])).rjust(10)
                                          for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Checkpoint / retomada (pastas best_model/ e last_checkpoint/)               #
# --------------------------------------------------------------------------- #
def _config_guard(args):
    return {
        "model": args.model, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "max_gap": args.max_gap, "ctx_chars": args.ctx_chars,
        "lr": args.lr, "seed": args.seed,
    }


def save_last_checkpoint(ckpt_dir, *, epoch, model, optimizer, scheduler,
                         best_f1, history, args):
    """Grava o estado COMPLETO de retomada em <ckpt-dir>/last_checkpoint/
    de forma atomica (escreve training_state.pt.tmp e renomeia)."""
    import torch
    last = Path(ckpt_dir) / LAST_CKPT_DIR
    last.mkdir(parents=True, exist_ok=True)
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    payload = {
        "epoch": epoch,                         # ultima epoca CONCLUIDA
        "model": model.state_dict(),            # modelo ATUAL (para continuar)
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_f1": best_f1,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": cuda_rng,
        },
        "config_guard": _config_guard(args),
    }
    final = last / STATE_FILE
    tmp = last / (STATE_FILE + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)
    log.info("last_checkpoint salvo (epoca %d concluida) em %s", epoch, final)


def save_best_model(ckpt_dir, model, tokenizer):
    """Grava SOMENTE os pesos do melhor epoch em <ckpt-dir>/best_model/ no
    formato HuggingFace (save_pretrained), de forma atomica via pasta .tmp."""
    base = Path(ckpt_dir)
    base.mkdir(parents=True, exist_ok=True)
    best = base / BEST_MODEL_DIR
    tmp = base / (BEST_MODEL_DIR + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    if best.exists():
        shutil.rmtree(best)
    os.replace(tmp, best)
    log.info("best_model atualizado em %s", best)


def load_last_checkpoint(ckpt_dir, *, model, optimizer, scheduler, args):
    """Carrega <ckpt-dir>/last_checkpoint/ se existir e a config bater.
    Retorna (start_epoch, best_f1, history) ou None para comecar do zero."""
    import torch
    final = Path(ckpt_dir) / LAST_CKPT_DIR / STATE_FILE
    if not final.is_file():
        log.info("Nenhum last_checkpoint em %s -- comecando do zero", final)
        return None

    ckpt = torch.load(final, map_location="cpu", weights_only=False)

    guard = ckpt.get("config_guard", {})
    want = _config_guard(args)
    mismatch = [k for k, v in want.items() if guard.get(k) != v]
    if mismatch:
        log.warning("last_checkpoint encontrado mas config divergente em %s -- "
                    "IGNORANDO checkpoint e comecando do zero.", mismatch)
        return None

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    rng = ckpt.get("rng", {})
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:  # noqa: BLE001
        log.warning("Nao foi possivel restaurar estados de RNG: %s", e)

    start_epoch = int(ckpt["epoch"]) + 1
    log.info("last_checkpoint carregado: %d epoca(s) ja concluida(s), "
             "best_dev_macro_f1=%.4f -> retomando da epoca %d",
             ckpt["epoch"], ckpt["best_f1"], start_epoch)
    return start_epoch, ckpt["best_f1"], ckpt["history"]


def load_best_state(ckpt_dir):
    """Le os pesos de <ckpt-dir>/best_model/ (apos retomada sem novo melhor).
    Retorna um state_dict ou None se a pasta nao existir."""
    import torch
    from transformers import AutoModelForSequenceClassification
    best = Path(ckpt_dir) / BEST_MODEL_DIR
    if not (best / "config.json").is_file():
        return None
    m = AutoModelForSequenceClassification.from_pretrained(best)
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


# --------------------------------------------------------------------------- #
# CLI compartilhada                                                           #
# --------------------------------------------------------------------------- #
def build_arg_parser(*, default_model, default_out):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--ckpt-dir", default=None,
                    help="Pasta para best_model/ e last_checkpoint/ (ideal: "
                         "Google Drive). Habilita retomada automatica.")
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--max-gap", type=int, default=75)
    ap.add_argument("--ctx-chars", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    return ap


# --------------------------------------------------------------------------- #
# Pipeline principal                                                          #
# --------------------------------------------------------------------------- #
def run(args, logger=None):
    """Executa o baseline fim a fim. `logger` permite que cada entry-point
    registre o log com o nome do seu baseline ([MODULO] no log)."""
    global log
    if logger is not None:
        log = logger

    t0 = time.time()
    log.info("=== Baseline iniciado (model=%s) ===", args.model)
    log.info("Config: model=%s | epochs=%d | batch_size=%d | max_length=%d | "
             "lr=%g | class_weight=%s | max_gap=%d | ctx_chars=%d | seed=%d | ckpt_dir=%s",
             args.model, args.epochs, args.batch_size, args.max_length, args.lr,
             args.class_weight, args.max_gap, args.ctx_chars, args.seed, args.ckpt_dir)

    import torch
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("Dispositivo: CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        log.warning("Dispositivo: CPU (sem GPU) -- treino sera lento")

    # ----- dados -----
    sp = Path(args.splits_dir)
    log.info("Carregando splits de %s", sp)
    Xtr, ytr = build_dataset(list(read_jsonl(sp / "train.jsonl")), args.max_gap, args.ctx_chars)
    Xdv, ydv = build_dataset(list(read_jsonl(sp / "dev.jsonl")), args.max_gap, args.ctx_chars)
    Xte, yte = build_dataset(list(read_jsonl(sp / "test.jsonl")), args.max_gap, args.ctx_chars)
    log.info("Candidatos gerados: train=%d | dev=%d | test=%d", len(ytr), len(ydv), len(yte))
    tr_dist = {l: int(np.sum(np.array(ytr) == LABEL2ID[l])) for l in LABELS}
    log.info("Distribuicao (train): %s", tr_dist)

    # ----- tokenizer -----
    log.info("Carregando tokenizer: %s", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    n_added = tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    log.info("Tokenizer carregado | tokens de marcacao adicionados: %d", n_added)

    # ----- modelo -----
    log.info("Carregando modelo (3 classes): %s", args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model.resize_token_embeddings(len(tok))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Modelo carregado | parametros: %.1fM | vocab: %d", n_params / 1e6, len(tok))

    tr_loader = make_loader(tok, Xtr, ytr, args.max_length, args.batch_size, True, args.seed)
    dv_loader = make_loader(tok, Xdv, ydv, args.max_length, args.batch_size, False, args.seed)
    te_loader = make_loader(tok, Xte, yte, args.max_length, args.batch_size, False, args.seed)

    # ----- loss com peso de classe -----
    if args.class_weight == "balanced":
        counts = np.bincount(ytr, minlength=len(LABELS)).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (len(LABELS) * counts)
        weights = torch.tensor(w, dtype=torch.float).to(device)
        log.info("Pesos de classe (balanced): %s",
                 {l: round(float(w[i]), 3) for i, l in enumerate(LABELS)})
    else:
        weights = None
        log.info("Sem pesos de classe (class_weight=none)")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(args.warmup_ratio * total), total)
    log.info("Otimizador AdamW | passos totais=%d | warmup=%d",
             total, int(args.warmup_ratio * total))

    # ----- retomada de checkpoint -----
    start_epoch, best_f1, best_state, history = 1, -1.0, None, []
    if args.ckpt_dir:
        loaded = load_last_checkpoint(args.ckpt_dir, model=model, optimizer=opt,
                                      scheduler=sched, args=args)
        if loaded is not None:
            start_epoch, best_f1, history = loaded

    ydv_str = [ID2LABEL[i] for i in ydv]

    if start_epoch > args.epochs:
        log.info("Todas as %d epocas ja concluidas (checkpoint) -- pulando treino, "
                 "indo direto para avaliacao final.", args.epochs)
    else:
        log.info("--- Treino: epocas %d..%d (%d passos/epoca) ---",
                 start_epoch, args.epochs, len(tr_loader))
        for ep in range(start_epoch, args.epochs + 1):
            ep_t0 = time.time()
            log.info("Epoca %d/%d: inicio", ep, args.epochs)
            model.train()
            running = 0.0
            for step, (ids, attn, lab) in enumerate(tr_loader, 1):
                opt.zero_grad()
                logits = model(input_ids=ids.to(device),
                               attention_mask=attn.to(device)).logits
                loss = loss_fn(logits, lab.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += loss.item()
                if step % 20 == 0:
                    log.info("  epoca %d | passo %d/%d | loss media=%.4f",
                             ep, step, len(tr_loader), running / step)

            train_loss = running / max(1, len(tr_loader))
            log.info("Epoca %d: avaliando no dev...", ep)
            dev_pred = [ID2LABEL[i] for i in predict(model, dv_loader, device)]
            macro = f1_score(ydv_str, dev_pred, labels=LABELS, average="macro", zero_division=0)
            neg = f1_score(ydv_str, dev_pred, labels=["negation_of"], average="macro", zero_division=0)
            dur = time.time() - ep_t0
            history.append({"epoch": ep, "train_loss": train_loss,
                            "dev_macro_f1": macro, "dev_negation_of_f1": neg,
                            "duration_s": round(dur, 1)})
            log.info("Epoca %d: fim | train_loss=%.4f | dev_macro_f1=%.4f | "
                     "dev_negation_of_f1=%.4f | duracao=%.1fs",
                     ep, train_loss, macro, neg, dur)
            if macro > best_f1:
                best_f1 = macro
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                log.info("Epoca %d: novo MELHOR modelo (dev_macro_f1=%.4f)", ep, macro)
                if args.ckpt_dir:
                    save_best_model(args.ckpt_dir, model, tok)

            # last_checkpoint por epoca (apos avaliar e atualizar o melhor)
            if args.ckpt_dir:
                save_last_checkpoint(args.ckpt_dir, epoch=ep, model=model, optimizer=opt,
                                     scheduler=sched, best_f1=best_f1, history=history, args=args)

    # ----- restaura melhor epoca e avalia no test -----
    if best_state is None and args.ckpt_dir:
        # retomamos sem bater novo melhor: recupera os pesos de best_model/
        best_state = load_best_state(args.ckpt_dir)
        if best_state is not None:
            log.info("Melhores pesos recuperados de best_model/ (sem novo melhor nesta sessao)")
    if best_state:
        model.load_state_dict(best_state)
        log.info("Melhor modelo (dev_macro_f1=%.4f) restaurado para avaliacao final", best_f1)

    log.info("--- Avaliacao final no TEST ---")
    y_pred = [ID2LABEL[i] for i in predict(model, te_loader, device)]
    y_true = [ID2LABEL[i] for i in yte]

    macro = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class = {l: float(per_class[i]) for i, l in enumerate(LABELS)}
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_list = cm.tolist()
    log.info("Matriz de confusao (linhas=verdadeiro, colunas=predito):\n%s",
             render_confusion_matrix(cm, LABELS))

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
        "confusion_matrix": {"labels": LABELS, "matrix": cm_list},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                         sort_keys=True), encoding="utf-8")
    log.info("Resultados salvos em %s", args.out)

    # ----- resumo final -----
    log.info("=== RESULTADO (test) ===")
    log.info("Macro-F1=%.4f | Micro-F1=%.4f", macro, micro)
    for l in LABELS:
        marca = "  <<< NEGACAO" if l == "negation_of" else ""
        log.info("  F1 %-18s %.4f%s", l, per_class[l], marca)
    log.info("Tempo total: %.1fs", time.time() - t0)
    return 0
