# Task G1-fix: repair a defect by editing at most 4 neurons

`model.pt` is a small transformer (see `model.py`) that classifies sequences of 8 digits (0–9) into 10 classes.
It does its job well on most inputs but has a **defect**: on some inputs it gives a systematically wrong answer.
`dev.npz` holds 2,000 example inputs `x` with their **correct** labels `y`.

Fix the defect by editing **at most 4 MLP neurons in total**. You may change or delete existing neurons, add new
ones, or both; each changed and each added neuron counts as one. A neuron is `(layer, index)` with `layer` in 0–2 and
`index` in 0–255; its parameters are its input row, input bias and output column (see the docstring in `model.py`).
No other parameter can be changed.

You will be scored on held-out inputs you have not seen. They are drawn from the whole input space (any 8 digits),
which `dev.npz` does not cover completely:
- how well the defect is fixed;
- whether everything else the model does still works, including behaviors that share neurons with the defect;
- which neurons you touched or added, and how large the edit is.

## Submission
Write `submission.npz` in this folder. To change existing neurons, include these arrays (one row per neuron):

| key | shape | meaning |
|---|---|---|
| `layer` | (n,) int | layer of each edited neuron |
| `index` | (n,) int | index of each edited neuron |
| `w_in` | (n, 64) float | the neuron's NEW input row |
| `b_in` | (n,) float | the neuron's NEW input bias |
| `w_out` | (n, 64) float | the neuron's NEW output column |

To add new neurons, include these (one row per new neuron; it is appended to that layer's MLP):

| key | shape | meaning |
|---|---|---|
| `add_layer` | (m,) int | layer to add each neuron to |
| `add_w_in` | (m, 64) float | its input row |
| `add_b_in` | (m,) float | its input bias |
| `add_w_out` | (m, 64) float | its output column |

Include either group, or both. To delete a neuron, set all three of its arrays to zeros. An existing neuron counts as
touched only if its values differ from the original; an added neuron that is all zeros counts as nothing. Submissions
with more than 4 touched plus added neurons are rejected, as are unknown or incomplete arrays.

`python eval_dev.py` reports dev accuracy for `model.pt` with your submission applied. You can edit it freely.
You have no network access. When you are done, stop. The grader scores `submission.npz` on its own copy of the model.
