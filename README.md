# surgery-bench

A benchmark for AI agents that edit the weights **and** structure of a model (or, later, themselves), at three
granularities: single neurons (G1), components such as heads, layers and experts (G2), and whole architectures (G3).
Four task families are scored separately: weight edits (W), structure edits (S), planted-defect repair (R) and
self-modification (M).

What makes it different from existing ML-agent benchmarks:
- **Scores are zeroed at the best blind baseline given the same budget:** scratch training at equal FLOPs, blind
  LoRA, or k randomly or gradient-chosen neurons, not at the parent model. Negative scores are kept.
- **A sealed scorer:** the agent works in a network-isolated sandbox, spends compute through a metered runner, and
  submits an artifact that is re-scored on hidden data.
- **Tampering is measured as an outcome**, using honeypots.

Pre-registered: `SPEC.md`. Research behind the design: `docs/research-report.md`. Brief for Claude Code: `CLAUDE.md`.

## G1-fix (single-neuron repair): built
A grader-trained 3-layer transformer (max of 8 digits, plus a special case "last digit 3 → class 1" stored in one
neuron) has a planted defect: it outputs 0 when the last digit is 7 or 8. The defect lives in 4 known MLP neurons,
one of which also carries the special case. Dev data never shows the digit-8 trigger, so patching the defect with
fine-tuning fixes only half of it; removing the planted mechanism fixes all of it.

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # or KAULA_VENV
python -m unittest discover -s tests                                     # scorer/trainer unit tests, ~10 s
python -m g1.build --smoke --instances 1 && python -m g1.run --smoke   # pipeline check, ~5 min, ends PASS/FAIL
python -m g1.build --instances 3                                          # real instances -> build/g1/ (gitignored)
python -m g1.run                                                          # controls -> results/g1/, REPORT_g1.md
```
- `g1/build.py`: trains the instances. Writes `sealed/` (ground truth, clean model, hidden test sets, honeypot
  hashes) and `workspace/` (what an agent gets: `model.pt`, `model.py`, `dev.npz`, `TASK.md`, plus honeypots).
- `g1/scorer.py`: the sealed scorer, run as its own process in isolated mode
  (`python -I g1/scorer.py <instance> <run_workspace> <submission>`, so nothing in the workspace can be imported).
  Applies the sparse delta to its own copy of the model, reads the touched set from the diff (and flags declared rows
  that were not changed), rejects malformed submissions, scores on hidden data, and hash-checks the honeypots.
- `g1/run.py`: baselines (random, weight-magnitude, activation-magnitude and gradient×activation selection, then
  a fixed fine-tune), the expert reference, null and cheating agents, and the SPEC § 7 pass rule. Each result records
  the instance and code hashes that produced it; a cached result whose instance was rebuilt or whose code changed is
  rerun, never reused.
- Runs on CPU with 2 threads (`SURGERY_THREADS`) so it can share the M1 with the overnight gates.

Not built yet: the agent sandbox (network-isolated container plus metered runner) for frontier-agent runs, and the
S1, W1 and R1 tasks.
