#!/usr/bin/env python3
"""
Geracao de pares-candidatos de relacao.

O SemClinBr so anota relacoes POSITIVAS. Para medir extracao de relacao (em
especial `negation_of`) precisamos tambem dos pares NEGATIVOS (`no_relation`):
pares de entidades para os quais nao ha relacao anotada.

Decisoes:
  1) Pares ORDENADOS (e1, e2): a direcao importa para `negation_of`.
  2) Janela maxima de caracteres entre os spans (`max_gap`, default 75) para
     nao explodir o numero de negativos.
  3) Nunca cruza documentos; descarta auto-pares e pares com span identico.
  4) Ordem deterministica por (start, end, id).
"""
from __future__ import annotations

from typing import Any, Iterator


def entity_gap(e1: dict, e2: dict) -> int:
    if e1["start"] <= e2["start"]:
        return max(0, e2["start"] - e1["end"])
    return max(0, e1["start"] - e2["end"])


def iter_candidate_pairs(doc: dict, max_gap: int = 75) -> Iterator[dict[str, Any]]:
    text = doc["text"]
    gold = {(r["e1_id"], r["e2_id"]): r["type"] for r in doc["relations"]}
    ents = sorted(doc["entities"], key=lambda e: (e["start"], e["end"], e["id"]))

    for i, e1 in enumerate(ents):
        for j, e2 in enumerate(ents):
            if i == j:
                continue
            if (e1["start"], e1["end"]) == (e2["start"], e2["end"]):
                continue
            if entity_gap(e1, e2) > max_gap:
                continue
            yield {
                "doc_id": doc["doc_id"],
                "e1": e1,
                "e2": e2,
                "label": gold.get((e1["id"], e2["id"]), "no_relation"),
            }
