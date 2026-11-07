"""Evaluate a model on the utility panel.

Usage: python -m panel.run --model DIR_OR_HF_ID [--parent DIR_OR_HF_ID] [--n N] [--ppl-tokens T] --out results.json
With --parent, also reports the internal canaries (KL to the parent on held-out text and on random tokens).
"""
import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import limit_threads, write_json_atomic  # noqa: E402
from panel.score import LM, kl_random, score_lambada, score_mc, score_text  # noqa: E402
from panel.tasks import MANIFEST, TASKS, load_items, load_manifest, to_request  # noqa: E402

WINDOW = 1024  # tokens per window for held-out NLL and KL


def load_hf(name_or_dir, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_dir)
    tok.model_max_length = 10**9  # held-out text is tokenized whole, then scored in windows
    return AutoModelForCausalLM.from_pretrained(name_or_dir, dtype=torch.float32).to(device).eval(), tok


def make_lm(model, tok, batch_size=16):
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    return LM(model, lambda s: tok(s, add_special_tokens=False)["input_ids"],
              max_length=min(getattr(model.config, "max_position_embeddings", 2048), 2048),
              batch_size=batch_size, pad_id=pad)


class PanelData:
    """The pinned items, loaded and checked once. n / ppl_tokens: use a prefix (for quick runs only)."""

    def __init__(self, n=None, ppl_tokens=None, tasks=None):
        self.manifest = load_manifest()
        self.tasks = list(tasks or TASKS)
        self.n, self.ppl_tokens = n, ppl_tokens or self.manifest["tasks"]["wikitext"]["tokens"]
        self.items = {t: load_items(t, self.manifest, n) for t in self.tasks}
        self._ids = {}

    def text_ids(self, lm):
        key = id(lm.encode)
        if key not in self._ids:
            self._ids[key] = lm.encode(self.items["wikitext"])[:self.ppl_tokens]
        return self._ids[key]

    def describe(self):
        with open(MANIFEST, "rb") as f:
            msha = hashlib.sha256(f.read()).hexdigest()
        return dict(manifest_sha=msha, n=self.n, ppl_tokens=self.ppl_tokens, tasks=self.tasks,
                    full_panel=self.n is None and self.ppl_tokens == self.manifest["tasks"]["wikitext"]["tokens"])


def evaluate(lm, data, parent=None):
    out = {}
    for name in data.tasks:
        kind = TASKS[name]["kind"]
        if kind == "mc":
            out[name] = score_mc(lm, [to_request(name, d) for d in data.items[name]])
        elif kind == "lambada":
            out[name] = score_lambada(lm, [to_request(name, d) for d in data.items[name]])
        else:
            out[name] = score_text(lm, data.text_ids(lm), WINDOW, parent)
    if parent is not None:
        out["kl_random"] = kl_random(lm, parent, lm.model.config.vocab_size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parent")
    ap.add_argument("--n", type=int)
    ap.add_argument("--ppl-tokens", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cpu, or mps on Apple silicon")
    a = ap.parse_args()
    limit_threads()
    data = PanelData(a.n, a.ppl_tokens)
    lm = make_lm(*load_hf(a.model, a.device))
    parent = make_lm(*load_hf(a.parent, a.device)) if a.parent else None
    write_json_atomic(a.out, dict(model=a.model, parent=a.parent, panel=data.describe(), results=evaluate(lm, data, parent)))


if __name__ == "__main__":
    main()
