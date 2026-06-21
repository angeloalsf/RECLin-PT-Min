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
teste; reportamos tambem Micro-F1, Macro-F1, F1 por classe, classification
report e matriz de confusao.

## Paridade entre os baselines

Os dois baselines compartilham **um unico nucleo** (`src/re_core.py`), entao sao
identicos por construcao em tudo, exceto o modelo:

- Entity markers / representacao de entrada: `[E1] ... [/E1] [E2] ... [/E2]`
  (tokens especiais) + janela de contexto.
- Loss: CrossEntropy com `class_weight=balanced`.
- Scheduler: linear com warmup; early stopping pela melhor epoca no **dev**.
- Salvamento de checkpoints: `best_model/` (pesos do melhor epoch, formato HF) e
  `last_checkpoint/` (estado completo de retomada), gravados de forma atomica.
- Metricas: Micro-F1, Macro-F1, F1 por classe, classification report, matriz de
  confusao.
- Seed de reprodutibilidade (42) e logging estruturado centralizado.

Trocar BioBERTpt por BERTimbau e so mudar `--model`, `--ckpt-dir` e `--out`.

## Estrutura

```
SemClinBr-xml-public-v1/   # XMLs do corpus (voce coloca aqui; licenca restrita)
src/
  parse_semclinbr.py       # XML -> data/processed/dataset.jsonl
  candidates.py            # gera pares-candidatos (inclui os negativos no_relation)
  make_splits.py           # splits 80/10/10 doc-level, seed 42
  re_core.py               # NUCLEO compartilhado: treino, metricas, checkpoints, logging
  baseline_biobertpt.py    # entry-point fino: BioBERTpt (clinico)
  baseline_bertimbau.py    # entry-point fino: BERTimbau (geral)
  utils/logger.py          # logging central -> terminal + logs/pipeline.log
data/
  processed/dataset.jsonl  # gerado
  splits/{train,dev,test}.jsonl
notebooks/
  baseline_biobertpt_colab.ipynb   # Colab T4: treina BioBERTpt com retomada
  baseline_bertimbau_colab.ipynb   # Colab T4: treina BERTimbau com retomada
results/{baseline_biobertpt,baseline_bertimbau}.json
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
```

ou simplesmente `bash run.sh`. No Colab (GPU T4), use os notebooks em
`notebooks/` — cada um clona o repo, monta o Drive, treina com **retomada
automatica** por epoca e (opcional) envia o modelo final para o Hugging Face Hub.

## Como interpretar

Compare `negation_of` e Micro-F1 dos dois `results/*.json`. Se o BioBERTpt
superar o BERTimbau de forma consistente, ha evidencia de que o pre-treino
clinico **importa** para esta tarefa; se empatarem, o pre-treino geral ja basta.

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

Hashes SHA dos splits, intervalos de confianca por bootstrap, analise de
variancia multi-seed e geracao de tabelas LaTeX. Sao adicoes diretas depois.
