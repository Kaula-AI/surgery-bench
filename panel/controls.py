"""Milestone-0 controls for the utility panel itself: does the damage guard measure damage?

On a real parent (SmolLM2-135M by default), before any SurgeryBench task uses the panel:
  negative        a no-op edit changes nothing and passes the gate.
  sensitivity     deleting a random fraction f of every layer's MLP neurons (2 seeds per f) lowers the established benchmarks
                  monotonically in f, and the gate flags the largest deletion. No task baselines exist yet, so the
                  gate here uses the floors alone.
  discrimination  deleting the least-contributing neurons of each layer (mean |activation| x output-weight norm) does
                  less damage than deleting as many per layer at random. A guard that can't tell a careful edit from a
                  careless one of the same size can't support SurgeryBench.
  growth          surgery can add as well as remove. Function-preserving additions (64 neurons per layer that write
                  nothing; 3 inserted identity layers) must leave every output unchanged (KL to the parent < 1e-6,
                  the SPEC's t=0 check for growth) and pass. Added neurons that do write into the residual stream
                  (output norm 0.3x, 1x, 3x the layer's mean) must damage more as the scale grows, and the gate must
                  flag the largest.
  validity        across all the edits, how well each internal canary (KL to the parent on random tokens and on
                  held-out text, held-out NLL) ranks the damage on the established benchmarks (Spearman, bootstrap CI).
                  Reported, not gated.

Usage: python -m panel.controls [--model HuggingFaceTB/SmolLM2-135M] [--device mps] [--n N] [--ppl-tokens T]
  -> results_panel/<model>/<arm>.json and REPORT_panel.md (REPORT_panel_reduced.md if --n or --ppl-tokens is given)
"""
import argparse
import copy
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import limit_threads, read_json, sha256_file, write_json_atomic  # noqa: E402
from panel import edits  # noqa: E402
from panel.gate import EXTERNAL, FLOORS, METRICS, change, external_damage, flatten, gate, spearman, spearman_ci  # noqa: E402
from panel.run import PanelData, evaluate, load_hf, make_lm  # noqa: E402
from panel.tasks import TASKS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRACTIONS = (0.01, 0.03, 0.1, 0.3)
SEEDS = (0, 1)
GROW_PER_LAYER = 64
GROW_SCALES = (0.3, 1.0, 3.0)
INSERT_AFTER = (7, 15, 23)
FUNCTION_PRESERVING_KL = 1e-6
CALIBRATION_TOKENS = 16384  # from WikiText-2 *train*, never the panel's test text
CODE = ("common.py", "panel/controls.py", "panel/edits.py", "panel/gate.py", "panel/run.py", "panel/score.py",
        "panel/tasks.py", "panel/manifest.json")
CANARIES = ("kl_random", "wikitext.kl_to_parent", "wikitext.nll", "wikitext.ece")


def arms():
    yield "parent", None
    yield "null", None
    for f in FRACTIONS:
        for s in SEEDS:
            yield f"random_f{f}_s{s}", ("random", f, s)
        yield f"careful_f{f}", ("careful", f, None)
    yield "grow_zero", ("grow", 0.0, None)
    yield "insert_zero", ("insert", None, None)
    for sc in GROW_SCALES:
        yield f"grow_s{sc}", ("grow", sc, None)


def calibration_ids(data, lm):
    import datasets
    e = data.manifest["tasks"]["wikitext"]
    train = datasets.load_dataset(e["repo"], e["config"], split="train", revision=e["revision"])
    return lm.encode("".join(train["text"][:4000]))[:CALIBRATION_TOKENS]


def run_arms(model_name, data, out_dir, device="cpu"):
    key = hashlib.sha256((model_name + repr(data.describe()) + "".join(sha256_file(os.path.join(ROOT, c)) for c in CODE))
                         .encode()).hexdigest()
    model, tok = load_hf(model_name, device)
    parent = make_lm(model, tok)
    magnitude = None
    res = {}
    for name, spec in arms():
        p = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(p) and read_json(p).get("key") == key:
            res[name] = read_json(p)
            continue
        m = copy.deepcopy(model) if name != "parent" else model
        kind = spec[0] if spec else None
        if kind == "grow":
            edits.grow_mlp(m, GROW_PER_LAYER, out_scale=spec[1])
            size = f"+{GROW_PER_LAYER * len(edits.mlps(m))} neurons"
        elif kind == "insert":
            edits.insert_identity_layers(m, INSERT_AFTER)
            size = f"+{len(INSERT_AFTER)} layers"
        else:
            neurons = []
            if kind == "random":
                neurons = edits.random_neurons(m, spec[1], spec[2])
            elif kind == "careful":
                if magnitude is None:
                    magnitude = edits.activation_magnitude(model, calibration_ids(data, parent))
                neurons = edits.least_contributing_neurons(m, spec[1], magnitude)
            edits.delete(m, neurons)
            size = f"-{len(neurons)} neurons" if neurons else "none"
        lm = parent if name == "parent" else make_lm(m, tok)
        r = dict(key=key, arm=name, spec=spec, edit=size, results=evaluate(lm, data, parent))
        write_json_atomic(p, r)
        res[name] = r
        print(f"{name}: {size}, " + ", ".join(f"{t} {r['results'][t].get('acc_norm', r['results'][t].get('acc', ''))}"
                                                        for t in TASKS if t != "wikitext"), flush=True)
    return res


def analyze(res):
    parent = flatten(res["parent"]["results"])
    ch = {a: change(flatten(r["results"]), parent) for a, r in res.items() if a != "parent"}
    dmg = {a: external_damage(c) for a, c in ch.items()}
    by_f = {f: [dmg[f"random_f{f}_s{s}"] for s in SEEDS] for f in FRACTIONS}
    mean_f = [float(np.mean(by_f[f])) for f in FRACTIONS]
    ok_null, _ = gate(ch["null"], FLOORS)
    ok_big, failed_big = gate(ch[f"random_f{FRACTIONS[-1]}_s{SEEDS[0]}"], FLOORS)
    grow = [dmg[f"grow_s{sc}"] for sc in GROW_SCALES]
    ok_grow_big, _ = gate(ch[f"grow_s{GROW_SCALES[-1]}"], FLOORS)
    preserving = {a: gate(ch[a], FLOORS)[0] and -ch[a]["wikitext.kl_to_parent"] < FUNCTION_PRESERVING_KL
                  and -ch[a]["kl_random"] < FUNCTION_PRESERVING_KL for a in ("grow_zero", "insert_zero")}
    edited = [a for a in ch if a not in ("null", "grow_zero", "insert_zero")]
    validity = {c: (spearman([-ch[a][c] for a in edited], [dmg[a] for a in edited]),
                    spearman_ci([-ch[a][c] for a in edited], [dmg[a] for a in edited])) for c in CANARIES}
    controls = {
        "negative: a no-op edit changes no metric and passes the gate":
            ok_null and all(abs(v) < 1e-9 for v in ch["null"].values()),
        "sensitivity: random deletion damages the established benchmarks monotonically in the fraction deleted":
            all(b >= a - 0.005 for a, b in zip(mean_f, mean_f[1:])) and mean_f[-1] > mean_f[0] + FLOORS[EXTERNAL[0]],
        f"sensitivity: the gate flags deleting {FRACTIONS[-1]:.0%} of neurons at random": not ok_big,
        "discrimination: deleting the least-contributing neurons of each layer damages less than deleting as many per layer at random":
            all(dmg[f"careful_f{f}"] < np.mean(by_f[f]) for f in FRACTIONS if np.mean(by_f[f]) > FLOORS[EXTERNAL[0]]),
        "growth: function-preserving additions (zero-output neurons, identity layers) leave every output unchanged and pass":
            all(preserving.values()),
        "growth: added neurons that write into the residual stream damage more as their scale grows, and the largest is flagged":
            all(b >= a - 0.005 for a, b in zip(grow, grow[1:])) and not ok_grow_big,
    }
    return dict(changes=ch, damage=dmg, mean_damage_by_fraction=dict(zip(map(str, FRACTIONS), mean_f)),
                damage_by_growth_scale=dict(zip(map(str, GROW_SCALES), grow)),
                gate_failures_largest=failed_big, validity=validity, controls=controls, parent=parent)


def write_report(a, res, data, model_name, path):
    d = data.describe()
    L = [f"# Utility panel controls{'' if d['full_panel'] else ' (REDUCED panel: not the official size)'}", "",
         f"Model `{model_name}`. Panel manifest `{d['manifest_sha'][:12]}`, items per task: {d['n'] or 'all pinned (1000)'}, "
         f"held-out tokens: {d['ppl_tokens']}.", "",
         f"**{'PASS' if all(a['controls'].values()) else 'FAIL'}**: the damage guard's controls.", ""]
    L += [f"- {'PASS' if ok else 'FAIL'}: {c}" for c, ok in a["controls"].items()]
    L += ["", "Parent: " + ", ".join(f"{m} {a['parent'][m]:.3f}" for m in EXTERNAL) +
          f", wikitext NLL {a['parent']['wikitext.nll']:.3f}, ECE {a['parent']['wikitext.ece']:.4f}.", "",
          "| arm | edit | " + " | ".join(METRICS) + " | external damage | gate |",
          "|---|---|" + "---|" * len(METRICS) + "---|---|"]
    for arm, c in a["changes"].items():
        ok, _ = gate(c, FLOORS)
        L.append(f"| {arm} | {res[arm]['edit']} | " + " | ".join(f"{c[m]:+.4f}" for m in METRICS) +
                 f" | {a['damage'][arm]:+.4f} | {'pass' if ok else 'FAIL'} |")
    L += ["", "Changes are signed so positive = better (KL canaries are minus the KL). External damage = mean accuracy "
          "lost over " + ", ".join(EXTERNAL) + ". The gate uses the floors only (" +
          ", ".join(sorted({f'{v}' for v in FLOORS.values()})) + "), since no task baselines exist yet.", "",
          "Validity: how well each internal canary ranks external damage across the edited models (Spearman, 95% CI).", ""]
    L += [f"- `{c}`: rho = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" for c, (rho, ci) in a["validity"].items()]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n", type=int)
    ap.add_argument("--ppl-tokens", type=int)
    ap.add_argument("--device", default="cpu", help="cpu, or mps on Apple silicon")
    a = ap.parse_args()
    limit_threads()
    data = PanelData(a.n, a.ppl_tokens)
    out = os.path.join(ROOT, "results_panel", a.model.replace("/", "__"))
    res = run_arms(a.model, data, out, a.device)
    an = analyze(res)
    write_json_atomic(os.path.join(out, "analysis.json"), an)
    path = os.path.join(ROOT, "REPORT_panel.md" if data.describe()["full_panel"] else "REPORT_panel_reduced.md")
    write_report(an, res, data, a.model, path)
    print("PASS" if all(an["controls"].values()) else "FAIL", "->", path)


if __name__ == "__main__":
    main()
