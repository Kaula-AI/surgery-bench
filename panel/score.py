"""Scoring a causal LM on the utility panel, plus the internal canaries (KL to the parent).

Conventions follow lm-evaluation-harness 0.4: no BOS token; a choice is scored by the summed log-probability of its
tokens after the prompt and a space; acc_norm divides by the choice's length in characters; LAMBADA's acc is whether
the greedy continuation equals the target word.
"""
import numpy as np
import torch
import torch.nn.functional as F


class LM:
    """Any model whose forward(input_ids=...) returns .logits, with an encode(text) -> token ids function."""

    def __init__(self, model, encode, max_length=2048, batch_size=16, pad_id=0):
        self.model, self.encode = model.eval(), encode
        self.max_length, self.batch_size, self.pad_id = max_length, batch_size, pad_id
        self.device = next(model.parameters()).device

    def encode_pair(self, context, continuation):
        n = len(context) - len(context.rstrip())  # trailing spaces move to the continuation, as in lm-eval
        if n:
            context, continuation = context[:-n], context[-n:] + continuation
        whole, ctx = self.encode(context + continuation), self.encode(context)
        return ctx, whole[len(ctx):]

    @torch.no_grad()
    def loglik(self, requests):
        """[(context, continuation)] -> [(summed log-prob of the continuation, whether it is the greedy one)]."""
        enc = [self.encode_pair(c, x) for c, x in requests]
        order = sorted(range(len(enc)), key=lambda i: -(len(enc[i][0]) + len(enc[i][1])))
        out = [None] * len(enc)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            seqs = [(enc[i][0] + enc[i][1])[-(self.max_length + 1):] for i in idx]
            inp = torch.full((len(seqs), max(len(s) for s in seqs) - 1), self.pad_id, dtype=torch.long)
            for r, s in enumerate(seqs):  # right padding: causal attention never sees it
                inp[r, :len(s) - 1] = torch.as_tensor(s[:-1])
            logp = F.log_softmax(self.model(input_ids=inp.to(self.device)).logits.float(), -1).cpu()
            for r, i in enumerate(idx):
                s, n = seqs[r], len(enc[i][1])
                tgt = torch.as_tensor(s[-n:])
                lp = logp[r, len(s) - 1 - n:len(s) - 1]
                out[i] = (float(lp.gather(1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
        return out

    @torch.no_grad()
    def window_logprobs(self, ids, window):
        """Log-probs of every next token over non-overlapping windows of `ids`: yields (log-probs, targets)."""
        for s in range(0, len(ids) - 1, window):
            chunk = torch.as_tensor(ids[s:s + window + 1])[None]
            if chunk.shape[1] < 2:
                break
            yield F.log_softmax(self.model(input_ids=chunk[:, :-1].to(self.device)).logits.float()[0], -1).cpu(), chunk[0, 1:]


def ece(probs, correct, bins=10):
    """Expected calibration error of the top-choice probability."""
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            total += m.mean() * abs(probs[m].mean() - correct[m].mean())
    return float(total)


def score_mc(lm, items):
    """items: [(context, [choices], gold)] -> acc, acc_norm, per-item acc_norm correctness.
    No calibration here: a softmax over summed choice log-likelihoods is dominated by choice length (HellaSwag's ECE
    came out at 0.63 on SmolLM2-135M), so calibration is measured on next-token predictions instead (score_text)."""
    reqs = [(ctx, " " + ch) for ctx, chs, _ in items for ch in chs]
    lls = iter(ll for ll, _ in lm.loglik(reqs))
    correct, correct_norm = [], []
    for _, chs, gold in items:
        ll = np.array([next(lls) for _ in chs])
        norm = ll / np.array([len(c) for c in chs], float)
        correct.append(int(ll.argmax() == gold))
        correct_norm.append(int(norm.argmax() == gold))
    return dict(acc=float(np.mean(correct)), acc_norm=float(np.mean(correct_norm)), n=len(items), per_item=correct_norm,
                per_item_acc=correct)


def score_lambada(lm, items):
    res = lm.loglik(items)
    return dict(acc=float(np.mean([g for _, g in res])), nll=float(-np.mean([ll for ll, _ in res])), n=len(items),
                per_item=[int(g) for _, g in res])


def score_text(lm, ids, window, parent=None):
    """On held-out text: mean next-token NLL (nats), calibration (ECE of the top-1 next-token probability against
    whether it is the actual next token), and KL(parent || model) per token."""
    nll, kl, n = 0.0, 0.0, 0
    top_p, top_ok = [], []
    parent_windows = parent.window_logprobs(ids, window) if parent is not None else None
    for logp, tgt in lm.window_logprobs(ids, window):
        nll -= float(logp.gather(1, tgt[:, None]).sum())
        best = logp.max(-1)
        top_p.append(best.values.exp().numpy())
        top_ok.append((best.indices == tgt).numpy())
        if parent_windows is not None:
            plogp, _ = next(parent_windows)
            kl += float((plogp.exp() * (plogp - logp)).sum())
        n += len(tgt)
    out = dict(nll=nll / n, ece=ece(np.concatenate(top_p), np.concatenate(top_ok)), tokens=n)
    if parent is not None:
        out["kl_to_parent"] = kl / n
    return out


@torch.no_grad()
def kl_random(lm, parent, vocab_size, n_seq=64, seq_len=64, seed=0):
    """KL(parent || model) per token on uniformly random token sequences: a canary generated by the benchmark, so no
    edit can have trained on it."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (n_seq, seq_len), generator=g)
    kl, n = 0.0, 0
    for b in range(0, n_seq, lm.batch_size):
        xb = x[b:b + lm.batch_size]
        lp = F.log_softmax(lm.model(input_ids=xb.to(lm.device)).logits.float(), -1).cpu()
        plp = F.log_softmax(parent.model(input_ids=xb.to(parent.device)).logits.float(), -1).cpu()
        kl += float((plp.exp() * (plp - lp)).sum())
        n += xb.numel()
    return kl / n
