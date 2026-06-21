#!/usr/bin/env bash
# Pipeline minimo RECLin-PT, fim a fim.
# Compara BioBERTpt (clinico) x BERTimbau (geral) na MESMA tarefa, mudando so o
# modelo -> responde "o pre-treinamento clinico importa para extracao de
# relacoes em textos medicos em portugues?".
set -e
cd "$(dirname "$0")"

# 1) Parse SemClinBr XML -> dataset.jsonl
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) Splits 80/10/10 (doc-level, seed 42)
python src/make_splits.py --input data/processed/dataset.jsonl \
    --out-dir data/splits

# 3a) Baseline CLINICO: BioBERTpt (precisa baixar o modelo; GPU recomendada)
python src/baseline_biobertpt.py --splits-dir data/splits \
    --epochs 3 --batch-size 16 \
    --out results/baseline_biobertpt.json

# 3b) Baseline GERAL: BERTimbau -- MESMOS hiperparametros (paridade total)
python src/baseline_bertimbau.py --splits-dir data/splits \
    --epochs 3 --batch-size 16 \
    --out results/baseline_bertimbau.json
