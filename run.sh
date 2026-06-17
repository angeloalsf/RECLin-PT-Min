#!/usr/bin/env bash
# Pipeline minimo RECLin-PT, fim a fim.
set -e
cd "$(dirname "$0")"

# 1) Parse SemClinBr XML -> dataset.jsonl
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) Splits 80/10/10 (doc-level, seed 42)
python src/make_splits.py --input data/processed/dataset.jsonl \
    --out-dir data/splits

# 3) Baseline BioBERTpt (precisa baixar o modelo; GPU recomendada)
python src/baseline_biobertpt.py --splits-dir data/splits \
    --model pucpr/biobertpt-all --epochs 3 --batch-size 16 \
    --out results/baseline_biobertpt.json
