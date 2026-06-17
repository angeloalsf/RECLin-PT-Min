# RECLin-PT (minimo)

Versao enxuta para medir se o **BioBERTpt** detecta bem relacoes de
**negacao** (`negation_of`) em notas clinicas do **SemClinBr**.

Espaco de rotulos (3 classes): `negation_of`, `associated_with`, `no_relation`.

## Estrutura

```
SemClinBr-xml-public-v1/   # XMLs do corpus (voce coloca aqui; licenca restrita)
src/
  parse_semclinbr.py       # XML -> data/processed/dataset.jsonl
  candidates.py            # gera pares-candidatos (inclui os negativos no_relation)
  make_splits.py           # splits 80/10/10 doc-level, seed 42
  baseline_biobertpt.py    # fine-tuning BioBERTpt + metrica (foco negation_of)
data/
  processed/dataset.jsonl  # gerado
  splits/{train,dev,test}.jsonl
run.sh                     # pipeline fim a fim
```

## Como rodar

```bash
pip install -r requirements.txt

# 1) parse
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) splits 80/10/10 (doc-level, seed 42)
python src/make_splits.py

# 3) baseline BioBERTpt (GPU recomendada; baixa o modelo da HF)
python src/baseline_biobertpt.py --epochs 3 --batch-size 16 \
    --out results/baseline_biobertpt.json
```

ou simplesmente `bash run.sh`.

## Decisoes principais

- **Split em nivel de documento** (nao de relacao): evita vazamento de
  vocabulario do mesmo prontuario entre train/test (leakage canonico do NLP
  clinico). Estratificado pela presenca de `negation_of` para garantir a
  classe-alvo nos tres splits. Seed fixo 42.
- **Candidatos negativos**: o SemClinBr so anota relacoes positivas. Para medir
  extracao de relacao precisamos gerar os pares `no_relation` (pares ordenados,
  janela de `max_gap=75` caracteres). Direcao importa para `negation_of`.
- **Baseline**: marcadores de entidade tipados `[E1]...[/E1] [E2]...[/E2]` +
  janela de contexto, encoder `pucpr/biobertpt-all`, cabeca de classificacao
  em 3 classes, CrossEntropy com `class_weight=balanced` (negativos dominam).
  Melhor epoca escolhida pelo macro-F1 no dev; teste reportado uma unica vez.
- **Metrica**: macro-F1, micro-F1 e **F1 por classe** -- o numero que responde
  "detecta bem negacao?" e o **F1 de `negation_of`** no teste.

## O que foi deixado de fora (de proposito)

Hashes SHA dos splits, intervalos de confianca por bootstrap, analise de
variancia multi-seed e geracao de tabelas LaTeX. Sao adicoes diretas depois.
