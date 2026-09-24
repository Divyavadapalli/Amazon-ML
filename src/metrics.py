"""Local re-implementation of the official leaderboard metric.

Macro F0.5 over Source-1 entities:
  * truth empty  -> 1.0 if prediction empty else 0.0   (stated in the problem statement)
  * truth non-empty, prediction empty -> 0.0            (not stated; assumed, precision undefined)
  * otherwise F0.5 = 1.25*P*R / (0.25*P + R), and 0.0 when there is no true positive.
"""
from typing import Dict, Iterable, Set

BETA2 = 0.25


def entity_f05(truth: Set[str], pred: Set[str]) -> float:
    """F0.5 for a single Source-1 entity."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(truth: Dict[str, Set[str]], pred: Dict[str, Set[str]],
              entities: Iterable[str] = None) -> dict:
    """Macro F0.5 over `entities` (default: every key of `truth`), plus diagnostics."""
    ents = list(entities) if entities is not None else list(truth)
    tot = 0.0
    n_single = single_ok = 0
    tp = fp = fn = 0
    for e in ents:
        t, p = truth.get(e, set()), pred.get(e, set())
        tot += entity_f05(t, p)
        if not t:
            n_single += 1
            single_ok += not p
        k = len(t & p)
        tp += k
        fp += len(p) - k
        fn += len(t) - k
    n = max(len(ents), 1)
    return {
        "f05_macro": tot / n,
        "n_entities": len(ents),
        "singleton_acc": single_ok / n_single if n_single else float("nan"),
        "pair_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "pair_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }


if __name__ == "__main__":
    # Worked example from the problem statement: expected 0.714
    print(round(entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}), 3))
