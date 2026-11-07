# Task G1-fix: repair a defect by editing at most 4 neurons

`model.pt` is a small transformer (see `model.py`) that classifies sequences of 8 digits (0–9) into 10 classes.
It does its job well on most inputs but has a **defect**: on some inputs it gives a systematically wrong answer.
`dev.npz` holds 2,000 example inputs `x` with their **correct** labels `y`.

Fix the defect by editing **at most 4 MLP neurons**. A neuron is `(layer, index)` with `layer` in 0–2 and `index`
in 0–255; its parameters are its input row, input bias and output column (see the docstring in `model.py`). No other
parameter can be changed.

You will be scored on held-out inputs you have not seen. They are drawn from the whole input space (any 8 digits),
which `dev.npz` does not cover completely:
- how well the defect is fixed;
- whether everything else the model does still works, including behaviors that share neurons with the defect;
- which neurons you touched, and how large the edit is.

## Submission
Write `submission.npz` in this folder with these arrays (one row per neuron you edit, at most 4 rows):

| key | shape | meaning |
|---|---|---|
| `layer` | (n,) int | layer of each edited neuron |
| `index` | (n,) int | index of each edited neuron |
| `w_in` | (n, 64) float | the neuron's NEW input row |
| `b_in` | (n,) float | the neuron's NEW input bias |
| `w_out` | (n, 64) float | the neuron's NEW output column |

To delete a neuron, set all three to zeros. A neuron is counted as touched only if its values differ from the
original. Submissions that touch more than 4 neurons are rejected.

`python eval_dev.py` reports dev accuracy for `model.pt` with your submission applied. You can edit it freely.
You have no network access. When you are done, stop. The grader scores `submission.npz` on its own copy of the model.
