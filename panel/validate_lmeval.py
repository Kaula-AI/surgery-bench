"""Does the panel measure what the established benchmarks measure? Same model, same items, compared item by item with
lm-evaluation-harness.

Runs lm-eval (0.4.x, Hugging Face backend) and the panel's scorer on the first N documents of each benchmark and
reports both scores and the share of items on which the two agree. Agreement should be ~100%: a gap means the panel
uses a different prompt, tokenization or scoring rule from the one behind the published numbers.

Needs lm-eval in the environment (pip install "lm-eval[hf]"); the rest of the panel does not.
Usage: python -m panel.validate_lmeval [--model HuggingFaceTB/SmolLM2-135M] [--limit 200] [--out FILE]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import limit_threads, write_json_atomic  # noqa: E402
from panel.run import load_hf, make_lm  # noqa: E402
from panel.score import score_lambada, score_mc  # noqa: E402
from panel.tasks import TASKS, load_manifest, load_split, to_request  # noqa: E402

LMEVAL = {"hellaswag": "hellaswag", "arc_easy": "arc_easy", "piqa": "piqa", "lambada": "lambada_openai"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out")
    a = ap.parse_args()
    limit_threads()
    import lm_eval
    res = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={a.model},dtype=float32",
                                  tasks=list(LMEVAL.values()), limit=a.limit, log_samples=True, batch_size=16,
                                  device="cpu")
    lm = make_lm(*load_hf(a.model))
    manifest = load_manifest()
    report = {}
    for name, le in LMEVAL.items():
        samples = sorted(res["samples"][le], key=lambda s: s["doc_id"])
        ids = [s["doc_id"] for s in samples]
        ds = load_split(name, manifest["tasks"][name]["revision"])
        reqs = [to_request(name, ds[i]) for i in ids]
        if TASKS[name]["kind"] == "mc":
            ours = score_mc(lm, reqs)
            pairs = {"acc": (ours["per_item_acc"], [s["acc"] for s in samples]),
                     "acc_norm": (ours["per_item"], [s["acc_norm"] for s in samples])}
        else:
            ours = score_lambada(lm, reqs)
            pairs = {"acc": (ours["per_item"], [s["acc"] for s in samples])}
        report[name] = {m: dict(panel=float(np.mean(o)), lm_eval=float(np.mean(t)),
                                agreement=float(np.mean(np.asarray(o) == np.asarray(t))), n=len(o))
                        for m, (o, t) in pairs.items()}
        for m, r in report[name].items():
            print(f"{name:10s} {m:8s} panel {r['panel']:.3f}  lm-eval {r['lm_eval']:.3f}  item agreement "
                  f"{r['agreement']:.1%} (n={r['n']})", flush=True)
    if a.out:
        write_json_atomic(a.out, dict(model=a.model, limit=a.limit, lm_eval_version=lm_eval.__version__, report=report))


if __name__ == "__main__":
    main()
