#!/usr/bin/env python3
"""
Baseline BioBERTpt (encoder CLINICO em portugues) para extracao de relacao.

Entry-point fino: toda a logica vive em `src/relation_extraction.py` (compartilhado com o
baseline BERTimbau). Aqui so escolhemos o modelo e os caminhos de saida, para
que a UNICA diferenca entre os dois experimentos seja o checkpoint de pre-treino
-- a condicao necessaria para responder, de forma valida:

    "O pre-treinamento clinico importa para extracao de relacoes em textos
     medicos em portugues?"

BioBERTpt (`pucpr/biobertpt-all`) e o lado CLINICO da comparacao; BERTimbau
(`neuralmind/bert-base-portuguese-cased`) e o lado GERAL.

Uso (CPU funciona, mas e lento; ideal GPU/Colab):
    python src/baseline_biobertpt.py --splits-dir data/splits \
        --epochs 3 --batch-size 32 \
        --ckpt-dir /content/drive/MyDrive/RECLin-PT-Min/checkpoints \
        --out results/baseline_biobertpt.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relation_extraction import build_arg_parser, run  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("baseline_biobertpt")

DEFAULT_MODEL = "pucpr/biobertpt-all"
DEFAULT_OUT = "results/baseline_biobertpt.json"


def main() -> int:
    ap = build_arg_parser(default_model=DEFAULT_MODEL, default_out=DEFAULT_OUT)
    args = ap.parse_args()
    return run(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
