# Utility panel controls

Model `HuggingFaceTB/SmolLM2-135M`. Panel manifest `731435bda4bf`, items per task: all pinned (1000), held-out tokens: 65536.

**PASS**: the damage guard's controls.

- PASS: negative: a no-op edit changes no metric and passes the gate
- PASS: sensitivity: random deletion damages the established benchmarks monotonically in the fraction deleted
- PASS: sensitivity: the gate flags deleting 30% of neurons at random
- PASS: discrimination: deleting the least-contributing neurons of each layer damages less than deleting as many per layer at random
- PASS: growth: function-preserving additions (zero-output neurons, identity layers) leave every output unchanged and pass
- PASS: growth: added neurons that write into the residual stream damage more as their scale grows, and the largest is flagged

Parent: hellaswag.acc_norm 0.447, arc_easy.acc_norm 0.578, piqa.acc_norm 0.664, lambada.acc 0.410, wikitext NLL 2.856, ECE 0.0043.

| arm | edit | hellaswag.acc_norm | arc_easy.acc_norm | piqa.acc_norm | lambada.acc | lambada.nll | wikitext.nll | wikitext.ece | wikitext.kl_to_parent | kl_random | external damage | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null | none | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | -0.0000 | -0.0000 | pass |
| random_f0.01_s0 | -450 neurons | -0.0020 | +0.0030 | -0.0070 | -0.0100 | -0.0783 | -0.0401 | -0.0057 | -0.0388 | -0.0433 | +0.0040 | FAIL |
| random_f0.01_s1 | -450 neurons | +0.0020 | +0.0090 | +0.0040 | -0.0160 | -0.1328 | -0.0551 | -0.0048 | -0.0622 | -0.1691 | +0.0002 | FAIL |
| careful_f0.01 | -450 neurons | -0.0030 | -0.0140 | -0.0070 | -0.0080 | -0.0607 | -0.0154 | -0.0025 | -0.0156 | -0.0197 | +0.0080 | FAIL |
| random_f0.03_s0 | -1380 neurons | -0.0080 | -0.0170 | -0.0090 | -0.0640 | -0.4376 | -0.1238 | -0.0029 | -0.1226 | -0.1702 | +0.0245 | FAIL |
| random_f0.03_s1 | -1380 neurons | -0.0080 | -0.0370 | -0.0100 | -0.0690 | -0.4908 | -0.2483 | -0.0044 | -0.2480 | -0.1828 | +0.0310 | FAIL |
| careful_f0.03 | -1380 neurons | -0.0030 | -0.0160 | -0.0040 | -0.0110 | -0.1748 | -0.0537 | -0.0035 | -0.0542 | -0.0712 | +0.0085 | FAIL |
| random_f0.1_s0 | -4620 neurons | -0.0500 | -0.0750 | -0.0540 | -0.1810 | -1.7508 | -0.9174 | -0.0227 | -0.8908 | -0.8905 | +0.0900 | FAIL |
| random_f0.1_s1 | -4620 neurons | -0.0490 | -0.1090 | -0.0380 | -0.2100 | -1.7881 | -0.6775 | -0.0052 | -0.6550 | -0.7855 | +0.1015 | FAIL |
| careful_f0.1 | -4620 neurons | -0.0250 | -0.0480 | -0.0390 | -0.0800 | -0.7070 | -0.2013 | -0.0063 | -0.1987 | -0.2141 | +0.0480 | FAIL |
| random_f0.3_s0 | -13830 neurons | -0.1640 | -0.2400 | -0.1530 | -0.3990 | -8.4512 | -3.5648 | -0.0456 | -3.5240 | -2.3378 | +0.2390 | FAIL |
| random_f0.3_s1 | -13830 neurons | -0.1430 | -0.2810 | -0.1460 | -0.4090 | -8.8865 | -2.9240 | -0.0142 | -2.8902 | -2.5845 | +0.2448 | FAIL |
| careful_f0.3 | -13830 neurons | -0.1240 | -0.1320 | -0.1240 | -0.1820 | -2.0114 | -0.8146 | -0.0114 | -0.8226 | -0.8484 | +0.1405 | FAIL |
| grow_zero | +1920 neurons | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | -0.0000 | -0.0000 | pass |
| insert_zero | +3 layers | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | -0.0000 | -0.0000 | pass |
| grow_s0.3 | +1920 neurons | +0.0020 | -0.0070 | +0.0060 | -0.0150 | -0.0683 | -0.0172 | -0.0005 | -0.0201 | -0.1093 | +0.0035 | FAIL |
| grow_s1.0 | +1920 neurons | -0.0120 | -0.0390 | -0.0050 | -0.1050 | -0.5930 | -0.1924 | -0.0038 | -0.1969 | -0.4279 | +0.0402 | FAIL |
| grow_s3.0 | +1920 neurons | -0.1680 | -0.2340 | -0.1510 | -0.4070 | -9.8011 | -3.8907 | -0.0612 | -3.8370 | -3.6276 | +0.2400 | FAIL |

Changes are signed so positive = better (KL canaries are minus the KL). External damage = mean accuracy lost over hellaswag.acc_norm, arc_easy.acc_norm, piqa.acc_norm, lambada.acc. The gate uses the floors only (0.01), since no task baselines exist yet.

Validity: how well each internal canary ranks external damage across the edited models (Spearman, 95% CI).

- `kl_random`: rho = 0.92 [0.64, 0.99]
- `wikitext.kl_to_parent`: rho = 0.92 [0.72, 0.98]
- `wikitext.nll`: rho = 0.92 [0.72, 0.98]
- `wikitext.ece`: rho = 0.77 [0.41, 0.95]
