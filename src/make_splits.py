#!/usr/bin/env python3
"""
Splits 80/10/10 em NIVEL DE DOCUMENTO (versao minima, sem SHA).

Por que nivel de documento e nao de relacao: relacoes do mesmo prontuario
compartilham vocabulario; se cairem em splits diferentes ha vazamento e a
metrica de teste infla. E o "leakage canonico" do NLP clinico.

Estratificacao leve por PRESENCA de `negation_of`: separamos os documentos que
tem ao menos uma relacao de negacao dos que nao tem, embaralhamos cada grupo
com seed 42 e cortamos 80/10/10 dentro de cada grupo. Isso garante negacao em
train/dev/test sem depender de pacote externo de estratificacao multi-rotulo.

Seed FIXO = 42 (reprodutibilidade). Sem hashes SHA por opcao de escopo.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n")
            n += 1
    return n


def split_indices(n, seed):
    """Indices 80/10/10 de uma lista ja embaralhada de tamanho n."""
    n_test = round(n * 0.10)
    n_dev = round(n * 0.10)
    test = list(range(0, n_test))
    dev = list(range(n_test, n_test + n_dev))
    train = list(range(n_test + n_dev, n))
    return train, dev, test


def summarize(name, docs):
    rels = Counter()
    with_neg = 0
    for d in docs:
        types = {r["type"] for r in d["relations"]}
        if "negation_of" in types:
            with_neg += 1
        for r in d["relations"]:
            rels[r["type"]] += 1
    return name, len(docs), with_neg, dict(rels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/dataset.jsonl")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    docs = sorted(read_jsonl(args.input), key=lambda d: int(d["doc_id"])
                  if d["doc_id"].isdigit() else d["doc_id"])
    if not docs:
        print("ERRO: dataset vazio")
        return 2

    has_neg = [d for d in docs if any(r["type"] == "negation_of"
                                      for r in d["relations"])]
    no_neg = [d for d in docs if d not in has_neg]

    rng = random.Random(args.seed)
    rng.shuffle(has_neg)
    rng.shuffle(no_neg)

    train, dev, test = [], [], []
    for group in (has_neg, no_neg):
        tr, dv, te = split_indices(len(group), args.seed)
        train += [group[i] for i in tr]
        dev += [group[i] for i in dv]
        test += [group[i] for i in te]

    # ordem deterministica final por doc_id
    keyf = lambda d: (int(d["doc_id"]) if d["doc_id"].isdigit() else 0, d["doc_id"])
    train.sort(key=keyf)
    dev.sort(key=keyf)
    test.sort(key=keyf)

    out = Path(args.out_dir)
    n_tr = write_jsonl(out / "train.jsonl", train)
    n_dv = write_jsonl(out / "dev.jsonl", dev)
    n_te = write_jsonl(out / "test.jsonl", test)

    n = len(docs)
    print("=" * 60)
    print(f"SPLITS 80/10/10 (doc-level, seed={args.seed})")
    print("=" * 60)
    print(f"  total docs: {n}")
    print(f"  train: {n_tr} ({100*n_tr/n:.1f}%)  "
          f"dev: {n_dv} ({100*n_dv/n:.1f}%)  test: {n_te} ({100*n_te/n:.1f}%)")
    print()
    for name, docs_ in (("train", train), ("dev", dev), ("test", test)):
        _, nd, wn, rels = summarize(name, docs_)
        neg = rels.get("negation_of", 0)
        asc = rels.get("associated_with", 0)
        print(f"  {name:<6s} docs={nd:<5d} docs_com_negacao={wn:<4d} "
              f"| negation_of={neg:<5d} associated_with={asc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
