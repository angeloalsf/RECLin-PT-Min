# RECLin-PT (minimo)

Versao enxuta para responder uma pergunta de pesquisa:

> **"O pre-treinamento clinico importa para extracao de relacoes em textos
> medicos em portugues?"**

Comparamos dois encoders em portugues na MESMA tarefa de extracao de relacao em
notas clinicas do **SemClinBr**, mudando **somente o checkpoint de pre-treino**:

- **BioBERTpt** (`pucpr/biobertpt-all`) — encoder **clinico**.
- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) — encoder **geral**
  (pre-treinado no brWaC).

Espaco de rotulos (3 classes): `negation_of`, `associated_with`, `no_relation`.
A metrica que responde "detecta bem negacao?" e o **F1 de `negation_of`** no
teste; reportamos tambem Micro-F1, Macro-F1, Weighted-F1, MCC, F1 por classe,
classification report e matriz de confusao -- e um **teste de significancia**
entre os dois modelos.

## Paridade entre os baselines

Os dois baselines compartilham **um unico nucleo** (`src/relation_extraction.py`),
entao sao identicos por construcao em tudo, exceto o modelo:

- Entity markers / representacao de entrada: `[E1] ... [/E1] [E2] ... [/E2]`
  (tokens especiais) + janela de contexto.
- Loss: CrossEntropy com `class_weight=balanced`.
- Scheduler: linear com warmup; early stopping pela melhor epoca no **dev**.
- Salvamento de checkpoints: `best_model/` (pesos do melhor epoch, formato HF) e
  `last_checkpoint/` (estado completo de retomada), gravados de forma atomica.
- Metricas: Micro-F1, Macro-F1, Weighted-F1, MCC, F1 por classe, classification
  report, matriz de confusao; curvas de treino/validacao (`train_loss` e
  `dev_loss` por epoca).
- Seed de reprodutibilidade (42) e logging estruturado centralizado.

Trocar BioBERTpt por BERTimbau e so mudar `--model`, `--ckpt-dir` e `--out`.

## Estrutura

```
SemClinBr-xml-public-v1/   # XMLs do corpus (voce coloca aqui; licenca restrita)
src/
  parse_semclinbr.py       # XML -> data/processed/dataset.jsonl
  candidates.py            # gera pares-candidatos (inclui os negativos no_relation)
  make_splits.py           # splits 80/10/10 doc-level, seed 42
  relation_extraction.py   # NUCLEO compartilhado: treino, metricas, checkpoints, logging
  baseline_biobertpt.py    # entry-point fino: BioBERTpt (clinico)
  baseline_bertimbau.py    # entry-point fino: BERTimbau (geral)
  significance.py          # McNemar + bootstrap pareado no F1 de negation_of
  utils/logger.py          # logging central -> terminal + logs/pipeline.log
data/
  processed/dataset.jsonl  # gerado
  splits/{train,dev,test}.jsonl
notebooks/
  baseline_biobertpt_colab.ipynb   # Colab T4: treina BioBERTpt com retomada
  baseline_bertimbau_colab.ipynb   # Colab T4: treina BERTimbau com retomada
  significance_colab.ipynb         # Colab CPU: McNemar + bootstrap (apos treinar os dois)
results/
  baseline_{biobertpt,bertimbau}.json          # metricas
  baseline_{biobertpt,bertimbau}.preds.json    # predicoes do test (para significancia)
  significance_*.json                          # relatorio do teste pareado
run.sh                     # pipeline fim a fim (parse -> splits -> os dois baselines)
```

## Como rodar (local)

```bash
pip install -r requirements.txt

# 1) parse
python src/parse_semclinbr.py --xml-dir SemClinBr-xml-public-v1 \
    --out data/processed/dataset.jsonl

# 2) splits 80/10/10 (doc-level, seed 42)
python src/make_splits.py

# 3a) baseline CLINICO (BioBERTpt)
python src/baseline_biobertpt.py --epochs 3 --batch-size 16 \
    --out results/baseline_biobertpt.json

# 3b) baseline GERAL (BERTimbau) -- MESMOS hiperparametros
python src/baseline_bertimbau.py --epochs 3 --batch-size 16 \
    --out results/baseline_bertimbau.json

# 4) significancia (usa os *.preds.json gerados em 3a/3b)
python src/significance.py \
    --a results/baseline_biobertpt.preds.json \
    --b results/baseline_bertimbau.preds.json \
    --target negation_of \
    --out results/significance_biobertpt_vs_bertimbau.json
```

ou simplesmente `bash run.sh` (passos 1-3). No Colab (GPU T4), use os notebooks
em `notebooks/`: rode primeiro os dois de treino (`baseline_*_colab.ipynb`) —
cada um clona o repo, monta o Drive, treina com **retomada automatica** por
epoca, plota as curvas, publica `results/*.preds.json` e (opcional) envia o
modelo final ao Hugging Face Hub. Depois rode `significance_colab.ipynb` (CPU,
sem GPU/Drive) para o teste pareado entre os dois.

## Metricas e como interpretar

- Compare o **F1 de `negation_of`** e o **Macro-F1** dos dois `results/*.json`.
  Micro-F1, Weighted-F1 e accuracy sao dominados por `no_relation` (~99% dos
  pares) e servem so de contexto -- nao sao a manchete.
- **MCC** (`test_mcc`) e um numero unico robusto a desbalanceamento.
- **Curvas**: `dev_history` traz `train_loss` e `dev_loss` por epoca (overfitting)
  e `dev_macro_f1`/`dev_negation_of_f1` (selecao de epoca).
- **Significancia** (`src/significance.py`): McNemar diz se os padroes de erro
  diferem; o bootstrap pareado da o intervalo de 95% e o p-valor da diferenca no
  F1 de `negation_of`. Se o IC95 nao cruza zero, a vantagem e significativa.

### Multiplas seeds (manual, no Colab)

Rode uma seed por execucao mudando `--seed`, `--out` e `--ckpt-dir` (para nao
colidir), e agregue offline:

```bash
python src/baseline_bertimbau.py --seed 43 \
    --ckpt-dir .../checkpoints_bertimbau_seed43 \
    --out results/baseline_bertimbau_seed43.json
```

```python
import json, glob, statistics as st
v = [json.load(open(f))["test_f1_per_class"]["negation_of"]
     for f in glob.glob("results/baseline_bertimbau_seed*.json")]
print(f"negation_of F1: {st.mean(v):.4f} ± {st.pstdev(v):.4f} (n={len(v)})")
```

## Decisoes principais

- **Split em nivel de documento** (nao de relacao): evita vazamento de
  vocabulario do mesmo prontuario entre train/test. Estratificado pela presenca
  de `negation_of`. Seed fixo 42.
- **Candidatos negativos**: o SemClinBr so anota relacoes positivas; geramos os
  pares `no_relation` (pares ordenados, janela `max_gap`). Direcao importa para
  `negation_of`.
- **Baseline**: marcadores de entidade tipados + janela de contexto, cabeca de
  classificacao em 3 classes, CrossEntropy com `class_weight=balanced`. Melhor
  epoca pelo macro-F1 no dev; teste reportado uma unica vez.

## O que foi deixado de fora (de proposito)

Hashes SHA dos splits, intervalos de confianca multi-seed automatizados e
geracao de tabelas LaTeX. Sao adicoes diretas depois.
