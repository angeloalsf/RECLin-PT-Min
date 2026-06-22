#!/usr/bin/env python3
"""
Teste de significancia entre dois baselines de extracao de relacao.

Responde, sobre o MESMO conjunto de teste: a diferenca observada entre dois
modelos (ex.: BioBERTpt clinico x BERTimbau geral) e real ou pode ser ruido da
amostra de teste? Usa as predicoes salvas por `relation_extraction.run` em
`<out>.preds.json` -- nao re-treina nada e nao usa GPU.

Dois testes complementares:

1) McNemar (pareado, por exemplo):
   Olha apenas os casos em que os modelos DISCORDAM no acerto: b = A acertou e
   B errou; c = A errou e B acertou. Sob a hipotese nula (mesma taxa de acerto),
   b ~ Binomial(b+c, 0.5). Usamos o teste binomial exato (scipy.binomtest), que
   nao precisa de correcao de continuidade nem aproximacao qui-quadrado. Mede se
   os PADROES DE ERRO globais diferem.

2) Bootstrap pareado no F1 de `negation_of` (a metrica-alvo):
   Reamostra os exemplos do teste com reposicao N vezes; em cada reamostra
   recalcula F1_A - F1_B para a classe `negation_of`. Reporta a diferenca media,
   o intervalo de confianca de 95% (percentis 2.5/97.5) e um p-valor bilateral
   (proporcao de reamostras cujo sinal contraria a diferenca observada). Mede
   diretamente se a vantagem no F1 da NEGACAO e robusta.

Ambos rodam em segundos (so reamostragem de arrays na CPU).

Uso:
    python src/significance.py \
        --a results/baseline_biobertpt.preds.json \
        --b results/baseline_bertimbau.preds.json \
        --target negation_of --n-boot 10000 --seed 42 \
        --out results/significance_biobertpt_vs_bertimbau.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger  # noqa: E402

log = get_logger("significance")


def load_preds(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d


def f1_for_class(y_true, y_pred, cls):
    """F1 da classe `cls` direto de arrays numpy (vetorizado, rapido)."""
    t = (y_true == cls)
    p = (y_pred == cls)
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def mcnemar_exact(correct_a, correct_b):
    """McNemar exato (binomial) sobre vetores booleanos de acerto."""
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))   # A certo, B errado
    c = int(np.sum(~correct_a & correct_b))   # A errado, B certo
    n = b + c
    if n == 0:
        return {"b_only_a_correct": b, "c_only_b_correct": c,
                "p_value": 1.0, "note": "sem discordancias"}
    p = binomtest(b, n, 0.5, alternative="two-sided").pvalue
    return {"b_only_a_correct": b, "c_only_b_correct": c,
            "n_discordant": n, "p_value": float(p)}


def paired_bootstrap(y_true, ya, yb, cls, n_boot, seed):
    """Bootstrap pareado da diferenca F1_A - F1_B na classe `cls`."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = f1_for_class(y_true, ya, cls) - f1_for_class(y_true, yb, cls)
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)             # reamostra COM reposicao
        yt, a, b = y_true[idx], ya[idx], yb[idx]
        diffs[k] = f1_for_class(yt, a, cls) - f1_for_class(yt, b, cls)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # p-valor bilateral: proporcao de reamostras com sinal contrario ao observado
    if obs >= 0:
        p = 2.0 * float(np.mean(diffs <= 0.0))
    else:
        p = 2.0 * float(np.mean(diffs >= 0.0))
    p = min(1.0, p)
    return {"observed_diff": float(obs), "mean_diff": float(np.mean(diffs)),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "p_value": p, "n_boot": int(n_boot)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="preds.json do modelo A (ex.: BioBERTpt)")
    ap.add_argument("--b", required=True, help="preds.json do modelo B (ex.: BERTimbau)")
    ap.add_argument("--target", default="negation_of", help="classe-alvo do bootstrap")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/significance.json")
    args = ap.parse_args()

    da, db = load_preds(args.a), load_preds(args.b)
    labels = da["labels"]
    if db["labels"] != labels:
        log.error("Labels divergentes entre A e B: %s vs %s", labels, db["labels"])
        return 2
    yt_a, yt_b = np.array(da["y_true"]), np.array(db["y_true"])
    if len(yt_a) != len(yt_b) or not np.array_equal(yt_a, yt_b):
        log.error("y_true de A e B nao batem (test diferente?). "
                  "A=%d exemplos, B=%d.", len(yt_a), len(yt_b))
        return 2

    y_true = yt_a
    ya, yb = np.array(da["y_pred"]), np.array(db["y_pred"])
    cls = labels.index(args.target)

    log.info("=== Significancia: A=%s  vs  B=%s ===", da.get("model"), db.get("model"))
    log.info("Exemplos no test: %d | classe-alvo do bootstrap: %s",
             len(y_true), args.target)

    # acuracias simples (contexto)
    acc_a = float(np.mean(ya == y_true))
    acc_b = float(np.mean(yb == y_true))
    f1a = f1_for_class(y_true, ya, cls)
    f1b = f1_for_class(y_true, yb, cls)
    log.info("Accuracy: A=%.4f | B=%.4f", acc_a, acc_b)
    log.info("F1(%s): A=%.4f | B=%.4f | A-B=%+.4f", args.target, f1a, f1b, f1a - f1b)

    mc = mcnemar_exact(ya == y_true, yb == y_true)
    log.info("McNemar (exato): b(A>B)=%d c(B>A)=%d | p=%.4g",
             mc.get("b_only_a_correct"), mc.get("c_only_b_correct"), mc["p_value"])

    bs = paired_bootstrap(y_true, ya, yb, cls, args.n_boot, args.seed)
    log.info("Bootstrap F1(%s): diff=%+.4f | IC95=[%+.4f, %+.4f] | p=%.4g | n=%d",
             args.target, bs["observed_diff"], bs["ci95_low"], bs["ci95_high"],
             bs["p_value"], bs["n_boot"])
    sig = "SIM" if (bs["ci95_low"] > 0 or bs["ci95_high"] < 0) else "NAO"
    log.info("IC95 exclui zero? %s -> diferenca em F1(%s) %s significativa (alpha=0.05)",
             sig, args.target, "E" if sig == "SIM" else "NAO E")

    out = {
        "model_a": da.get("model"), "model_b": db.get("model"),
        "n_test": int(len(y_true)), "target_class": args.target,
        "accuracy": {"a": acc_a, "b": acc_b},
        "target_f1": {"a": f1a, "b": f1b, "a_minus_b": f1a - f1b},
        "mcnemar": mc,
        "paired_bootstrap": bs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                         sort_keys=True), encoding="utf-8")
    log.info("Relatorio de significancia salvo em %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
