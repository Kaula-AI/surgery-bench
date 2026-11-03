# Working brief for Claude Code: surgery-bench

You are building a benchmark that scores how well an AI agent edits the **weights and structure** of a model, from
single neurons up to whole-architecture changes. Tyler's machine is an M1 Mac with 16 GB; bigger tiers run on rented
RunPod GPUs. Read `SPEC.md` first: it fixes tasks, metrics, baselines and pass rules. `docs/research-report.md` is
the research behind it. **Never change a primary metric, baseline, budget or pass rule after agent results exist.**
Changes go under "Amendments" in SPEC.md with a date and reason.

## Principles
- The hard part is the baseline. Every score is zeroed at the best compute- or edit-size-matched baseline.
- The scorer is sealed. The agent never shares a process with anything that measures it, and the scorer re-scores the
  submitted artifact from scratch on hidden data.
- Milestone 0 tests the harness (positive, negative and seal controls) before any agent result counts.

## Build order (milestone 0)
1. Measure M1 throughput (FLOP/s on MPS for the parent sizes) and record the budgets under SPEC.md → Amendments.
2. G1-fix: **built** (`g1/`, see README). Smoke: `python -m g1.build --smoke --instances 1 && python -m g1.run --smoke`.
   Real: `python -m g1.build --instances 3 && python -m g1.run`, which writes `REPORT_g1.md`. Report its PASS/FAIL line as written.
3. Agent sandbox: network-isolated container, metered runner, workspace mount only (never `sealed/`), then score
   the run's `submission.npz` with `python -I g1/scorer.py <instance> <run_workspace> <submission>` (isolated mode:
   nothing in the agent-controlled workspace can be imported, whatever the cwd).
4. S1-grow, W1-facts, R1-circuit.
5. Controls from SPEC § 7, then agent runs.

## Rules
- Share the M1 with the overnight gates: check `../overnight.pid` and don't launch heavy jobs while it runs.
- Synthetic, benchmark-generated targets only. No real-user data.
- Bug fix after results exist: fix, delete the affected results, rerun, add an Amendment.
