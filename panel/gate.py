"""The utility-panel retention gate and the validity statistics (SPEC amendment 2026-11-07, "utility panel").

Every metric is turned into a signed change against the parent, positive = better, so one rule covers all of them:
an edit passes if, for every metric, change >= -max(2 * sigma_b, floor), where sigma_b is the seed standard deviation
of that metric's change pooled over the task's baseline arms (as in G1's retention gate) and the floor covers item
noise when the baselines barely move it.
"""
import numpy as np

HIGHER_IS_BETTER = ("hellaswag.acc_norm", "arc_easy.acc_norm", "piqa.acc_norm", "lambada.acc")
LOWER_IS_BETTER = ("lambada.nll", "wikitext.nll", "wikitext.ece")
KL_CANARIES = ("wikitext.kl_to_parent", "kl_random")  # the parent's value is 0 by definition
METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER + KL_CANARIES
# Floors: 1 accuracy point; 0.01 in ECE; 0.01 nats per token in NLL and KL.
FLOORS = {m: 0.01 for m in METRICS}
EXTERNAL = HIGHER_IS_BETTER  # the established-benchmark accuracies


def flatten(results):
    f = {}
    for task, r in results.items():
        if isinstance(r, dict):
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    f[f"{task}.{k}"] = float(v)
        else:
            f[task] = float(r)
    return f


def change(edit, parent):
    """Signed change of every gated metric, positive = better. Both arguments are flattened results."""
    out = {}
    for m in HIGHER_IS_BETTER:
        out[m] = edit[m] - parent[m]
    for m in LOWER_IS_BETTER:
        out[m] = parent[m] - edit[m]
    for m in KL_CANARIES:
        out[m] = -edit[m]
    return out


def tolerances(baseline_changes):
    """baseline_changes: {arm: [change dict per seed]} -> {metric: tolerance}."""
    tol = {}
    for m in METRICS:
        v = [np.var([c[m] for c in cs], ddof=1) for cs in baseline_changes.values() if len(cs) > 1]
        tol[m] = max(2 * float(np.sqrt(np.mean(v))) if v else 0.0, FLOORS[m])
    return tol


def gate(ch, tol):
    """(passed, {metric: (change, tolerance)} for the metrics that failed)."""
    failed = {m: (ch[m], tol[m]) for m in METRICS if ch[m] < -tol[m]}
    return not failed, failed


def external_damage(ch):
    """Mean accuracy lost on the established benchmarks, in points (0.01 = one point)."""
    return float(-np.mean([ch[m] for m in EXTERNAL]))


def _ranks(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y, n=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    s = [spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    s = np.array([v for v in s if np.isfinite(v)])
    return (float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))) if len(s) else (float("nan"),) * 2
