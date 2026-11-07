"""The utility panel's tasks: established benchmarks, each pinned to a dataset revision and a fixed item subset.

The panel is a damage guard (the "held-out utility panel" of SPEC section 5), not a target: an edit counts only if the
edited model still does about as well on these as blind baselines do. Prompts and scoring follow lm-evaluation-harness
0.4 so the numbers mean what the published ones mean (checked item by item: see panel/validate_lmeval.py).

Grader side only: the panel's data and its cache are never mounted in an agent's sandbox.
"""
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")
SUBSET_SEED = "surgery-bench/panel-v1"

# kind "mc": multiple choice scored by log-likelihood of each choice (acc, acc_norm, calibration).
# kind "lambada": last-word prediction (acc = greedy match, plus the target's NLL).
# kind "ppl": token-level NLL of held-out text, and the KL to the parent on the same text.
TASKS = {
    "hellaswag": dict(repo="Rowan/hellaswag", config=None, split="validation", kind="mc"),
    "arc_easy": dict(repo="allenai/ai2_arc", config="ARC-Easy", split="test", kind="mc"),
    "piqa": dict(repo="baber/piqa", config=None, split="validation", kind="mc"),
    "lambada": dict(repo="EleutherAI/lambada_openai", config="default", split="test", kind="lambada"),
    "wikitext": dict(repo="Salesforce/wikitext", config="wikitext-2-raw-v1", split="test", kind="ppl"),
}
# Official panel size, pinned in the manifest. A run may use a prefix (a prefix of a random permutation is itself a
# random subset), but a reported gate decision uses the full pinned subset.
N_ITEMS = 1000
PPL_TOKENS = 65_536


def _hellaswag_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def to_request(name, doc):
    """lm-eval 0.4 prompt for one item: (context, [choices], gold) for mc, (context, target) for lambada."""
    if name == "hellaswag":
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return (_hellaswag_pre(doc["activity_label"] + ": " + ctx), [_hellaswag_pre(e) for e in doc["endings"]],
                int(doc["label"]))
    if name == "arc_easy":
        return ("Question: " + doc["question"] + "\nAnswer:", list(doc["choices"]["text"]),
                list(doc["choices"]["label"]).index(doc["answerKey"]))
    if name == "piqa":
        return "Question: " + doc["goal"] + "\nAnswer:", [doc["sol1"], doc["sol2"]], int(doc["label"])
    if name == "lambada":
        words = doc["text"].split(" ")
        return " ".join(words[:-1]), " " + words[-1]
    raise ValueError(name)


def item_hash(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def load_split(name, revision):
    import datasets  # grader-side dependency only
    t = TASKS[name]
    return datasets.load_dataset(t["repo"], t["config"], split=t["split"], revision=revision)


def pin(n_items=N_ITEMS):
    """Build the manifest: each task's dataset revision and a fixed random subset of items. Run once; changing the
    manifest changes the panel, so it needs a dated SPEC amendment."""
    from huggingface_hub import HfApi
    api = HfApi()
    seed = int.from_bytes(hashlib.sha256(SUBSET_SEED.encode()).digest()[:8], "big")
    out = dict(version=1, subset_seed=SUBSET_SEED, lm_eval_conventions="0.4", tasks={})
    for name, t in TASKS.items():
        rev = api.dataset_info(t["repo"]).sha
        ds = load_split(name, rev)
        entry = dict(repo=t["repo"], config=t["config"], split=t["split"], revision=rev, kind=t["kind"], size=len(ds))
        if t["kind"] == "ppl":
            text = "".join(ds["text"])
            entry.update(text_sha256=hashlib.sha256(text.encode()).hexdigest(), tokens=PPL_TOKENS)
        else:
            task_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
            idx = np.random.default_rng([seed, task_seed]).permutation(len(ds))[:n_items].tolist()
            entry.update(indices=idx, items_sha256=hashlib.sha256("".join(item_hash(ds[i]) for i in idx).encode()).hexdigest())
        out["tasks"][name] = entry
    with open(MANIFEST, "w") as f:
        json.dump(out, f, indent=1)
    return out


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def load_items(name, manifest, n=None):
    """The pinned items of one task, checked against the manifest's hashes. n: use only the first n."""
    e = manifest["tasks"][name]
    ds = load_split(name, e["revision"])
    if e["kind"] == "ppl":
        text = "".join(ds["text"])
        if hashlib.sha256(text.encode()).hexdigest() != e["text_sha256"]:
            raise RuntimeError(f"{name}: text differs from the pinned manifest")
        return text
    docs = [ds[i] for i in e["indices"]]
    if hashlib.sha256("".join(item_hash(d) for d in docs).encode()).hexdigest() != e["items_sha256"]:
        raise RuntimeError(f"{name}: items differ from the pinned manifest")
    return docs[:n] if n else docs


if __name__ == "__main__":
    m = pin()
    print({k: (v["revision"][:10], v.get("size")) for k, v in m["tasks"].items()})
