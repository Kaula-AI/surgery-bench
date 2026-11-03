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

Status: spec only. No harness code yet.
