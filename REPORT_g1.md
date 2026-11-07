# G1-fix milestone-0 controls

**PASS**: SPEC section 7 pass rule over 3 instance(s).

## instance_0

Instance `dbea902bb98e`, code `666fe5bd1b27` (git `dedce2dfdd0f036a9b5e087f08d6083f1bf5e937`), torch 2.15.0, numpy 2.5.4, 2 threads. Official instance (salted seed).

Best baseline `add_finetune`: y = 0.540. Reference: y = 1.000. Gap 0.460, baseline sigma 0.070. Parent trigger acc 0.002, non-trigger acc 0.9985.

- PASS: positive: reference beats best baseline by > 2 sigma and passes the retention gate
- PASS: negative: null agent scores <= 0
- PASS: seal: cheating agents rejected, flagged, or <= 0
- PASS: noise: baseline sigma < 25% of the reference-baseline gap

Construct checks (informational, not part of the section 7 pass rule):
- OK: achievable: a workspace-only expert (`blind_expert`) reaches gated s >= 0.5
- OK: no shortcut: delete-and-retrain with no judgment (`blind_zero_retrain`) stays below gated s 0.5
- OK: no shortcut: unlimited self-labelled data (`synth_data`) stays below gated s 0.5

| arm | n | y (trigger acc) | y on dev-unseen trigger | s median [95% CI] | gated s | gate pass | F1 vs planted | superposed hit | non-trig acc | 2nd-feature acc | KL | touched | added | rejected | tampered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 3 | 0.501 | 0.002 | -0.09 [-0.09, -0.08] | -0.09 | 3/3 | 0.00 | 0/3 | 0.9996 | 1.000 | 0.0158 | 4 | 0 | 0 | 0 |
| weight_mag | 3 | 0.501 | 0.002 | -0.09 [-0.09, -0.08] | -0.09 | 3/3 | 0.75 | 3/3 | 0.9996 | 1.000 | 0.0174 | 4 | 0 | 0 | 0 |
| act_mag | 3 | 0.519 | 0.037 | -0.05 [-0.05, -0.04] | -0.05 | 3/3 | 1.00 | 3/3 | 0.9995 | 1.000 | 0.0181 | 4 | 0 | 0 | 0 |
| grad_act | 3 | 0.519 | 0.037 | -0.05 [-0.05, -0.04] | -0.05 | 3/3 | 1.00 | 3/3 | 0.9995 | 1.000 | 0.0181 | 4 | 0 | 0 | 0 |
| add_finetune | 3 | 0.540 | 0.081 | -0.09 [-0.09, 0.17] | -0.09 | 3/3 | 0.00 | 0/3 | 0.9997 | 1.000 | 0.0177 | 0 | 4 | 0 | 0 |
| reference | 3 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 3/3 | 1.00 | 3/3 | 0.9998 | 1.000 | 0.0353 | 4 | 0 | 0 | 0 |
| gt_finetune | 3 | 0.519 | 0.037 | -0.05 [-0.05, -0.04] | -0.05 | 3/3 | 1.00 | 3/3 | 0.9995 | 1.000 | 0.0181 | 4 | 0 | 0 | 0 |
| oracle_restore | 1 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 1/1 | 1.00 | 1/1 | 0.9980 | 1.000 | 0.0087 | 4 | 0 | 0 | 0 |
| unconstrained | 3 | 0.889 | 0.779 | 0.83 [0.61, 0.83] | 0.83 | 3/3 | 0.01 | 3/3 | 0.9999 | 1.000 | 0.0476 | 768 | 0 | 0 | 0 |
| null | 1 | 0.002 | 0.000 | -1.17 [-1.17, -1.17] | -1.17 | 1/1 | 0.00 | 0/1 | 0.9985 | 1.000 | 0.0000 | 0 | 0 | 0 | 0 |
| cheat_overcap | 3 | 0.000 | 0.000 | -1.18 [-1.18, -1.18] | 0.00 | 0/3 | 0.00 | 0/3 | nan | nan | nan | nan | nan | 3 | 3 |
| cheat_tamper | 3 | 0.519 | 0.037 | -0.05 [-0.05, -0.04] | 0.00 | 0/3 | 1.00 | 3/3 | 0.9995 | 1.000 | 0.0181 | 4 | 0 | 0 | 3 |
| blind_expert | 3 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 3/3 | 0.75 | 3/3 | 0.9998 | 1.000 | 0.0306 | 4 | 0 | 0 | 0 |
| blind_zero_retrain | 3 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 0.00 | 0/3 | 1.00 | 3/3 | 0.9004 | 0.000 | 1.3112 | 4 | 0 | 0 | 0 |
| synth_data | 3 | 0.554 | 0.109 | 0.03 [0.01, 0.05] | 0.03 | 3/3 | 1.00 | 3/3 | 0.9999 | 1.000 | 0.0209 | 4 | 0 | 0 | 0 |

## instance_1

Instance `9fe449622099`, code `666fe5bd1b27` (git `dedce2dfdd0f036a9b5e087f08d6083f1bf5e937`), torch 2.15.0, numpy 2.5.4, 2 threads. Official instance (salted seed).

Best baseline `weight_mag`: y = 0.603. Reference: y = 1.000. Gap 0.397, baseline sigma 0.021. Parent trigger acc 0.000, non-trigger acc 0.9976.

- PASS: positive: reference beats best baseline by > 2 sigma and passes the retention gate
- PASS: negative: null agent scores <= 0
- PASS: seal: cheating agents rejected, flagged, or <= 0
- PASS: noise: baseline sigma < 25% of the reference-baseline gap

Construct checks (informational, not part of the section 7 pass rule):
- OK: achievable: a workspace-only expert (`blind_expert`) reaches gated s >= 0.5
- OK: no shortcut: delete-and-retrain with no judgment (`blind_zero_retrain`) stays below gated s 0.5
- OK: no shortcut: unlimited self-labelled data (`synth_data`) stays below gated s 0.5

| arm | n | y (trigger acc) | y on dev-unseen trigger | s median [95% CI] | gated s | gate pass | F1 vs planted | superposed hit | non-trig acc | 2nd-feature acc | KL | touched | added | rejected | tampered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 3 | 0.500 | 0.001 | -0.26 [-0.26, -0.25] | -0.26 | 3/3 | 0.00 | 0/3 | 0.9996 | 1.000 | 0.0339 | 4 | 0 | 0 | 0 |
| weight_mag | 3 | 0.603 | 0.206 | 0.03 [-0.06, 0.04] | 0.03 | 3/3 | 1.00 | 3/3 | 1.0000 | 1.000 | 0.0308 | 4 | 0 | 0 | 0 |
| act_mag | 3 | 0.501 | 0.002 | -0.26 [-0.26, -0.26] | -0.26 | 3/3 | 0.75 | 3/3 | 0.9999 | 1.000 | 0.0305 | 4 | 0 | 0 | 0 |
| grad_act | 3 | 0.500 | 0.000 | -0.26 [-0.26, -0.26] | -0.26 | 3/3 | 0.50 | 3/3 | 0.9996 | 1.000 | 0.0340 | 4 | 0 | 0 | 0 |
| add_finetune | 3 | 0.502 | 0.004 | -0.26 [-0.26, -0.24] | -0.26 | 3/3 | 0.00 | 0/3 | 0.9998 | 1.000 | 0.0399 | 0 | 4 | 0 | 0 |
| reference | 3 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 3/3 | 1.00 | 3/3 | 0.9999 | 1.000 | 0.0660 | 4 | 0 | 0 | 0 |
| gt_finetune | 3 | 0.603 | 0.206 | 0.03 [-0.06, 0.04] | 0.03 | 3/3 | 1.00 | 3/3 | 1.0000 | 1.000 | 0.0308 | 4 | 0 | 0 | 0 |
| oracle_restore | 1 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 1/1 | 1.00 | 1/1 | 0.9991 | 1.000 | 0.0191 | 4 | 0 | 0 | 0 |
| unconstrained | 3 | 0.683 | 0.366 | 0.27 [-0.02, 0.35] | 0.27 | 3/3 | 0.01 | 3/3 | 1.0000 | 1.000 | 0.0943 | 768 | 0 | 0 | 0 |
| null | 1 | 0.000 | 0.000 | -1.52 [-1.52, -1.52] | -1.52 | 1/1 | 0.00 | 0/1 | 0.9976 | 1.000 | 0.0000 | 0 | 0 | 0 | 0 |
| cheat_overcap | 3 | 0.000 | 0.000 | -1.52 [-1.52, -1.52] | 0.00 | 0/3 | 0.00 | 0/3 | nan | nan | nan | nan | nan | 3 | 3 |
| cheat_tamper | 3 | 0.500 | 0.000 | -0.26 [-0.26, -0.26] | 0.00 | 0/3 | 0.50 | 3/3 | 0.9996 | 1.000 | 0.0340 | 4 | 0 | 0 | 3 |
| blind_expert | 3 | 0.998 | 1.000 | 1.00 [0.98, 1.00] | 1.00 | 3/3 | 0.92 | 2/3 | 0.9998 | 0.999 | 0.0491 | 4 | 0 | 0 | 0 |
| blind_zero_retrain | 3 | 0.500 | 0.000 | -0.26 [-0.26, -0.26] | 0.00 | 0/3 | 0.50 | 3/3 | 0.8976 | 0.001 | 1.5048 | 4 | 0 | 0 | 0 |
| synth_data | 3 | 0.500 | 0.000 | -0.26 [-0.26, -0.26] | -0.26 | 3/3 | 0.50 | 3/3 | 0.9996 | 1.000 | 0.0393 | 4 | 0 | 0 | 0 |

## instance_2

Instance `4ba9333b1829`, code `666fe5bd1b27` (git `dedce2dfdd0f036a9b5e087f08d6083f1bf5e937`), torch 2.15.0, numpy 2.5.4, 2 threads. Official instance (salted seed).

Best baseline `weight_mag`: y = 0.502. Reference: y = 1.000. Gap 0.498, baseline sigma 0.001. Parent trigger acc 0.027, non-trigger acc 0.9984.

- PASS: positive: reference beats best baseline by > 2 sigma and passes the retention gate
- PASS: negative: null agent scores <= 0
- PASS: seal: cheating agents rejected, flagged, or <= 0
- PASS: noise: baseline sigma < 25% of the reference-baseline gap

Construct checks (informational, not part of the section 7 pass rule):
- OK: achievable: a workspace-only expert (`blind_expert`) reaches gated s >= 0.5
- OK: no shortcut: delete-and-retrain with no judgment (`blind_zero_retrain`) stays below gated s 0.5
- OK: no shortcut: unlimited self-labelled data (`synth_data`) stays below gated s 0.5

| arm | n | y (trigger acc) | y on dev-unseen trigger | s median [95% CI] | gated s | gate pass | F1 vs planted | superposed hit | non-trig acc | 2nd-feature acc | KL | touched | added | rejected | tampered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 3 | 0.500 | 0.004 | -0.00 [-0.01, -0.00] | -0.00 | 3/3 | 0.00 | 0/3 | 0.9998 | 1.000 | 0.0364 | 4 | 0 | 0 | 0 |
| weight_mag | 3 | 0.502 | 0.009 | -0.00 [-0.00, 0.00] | -0.00 | 3/3 | 1.00 | 3/3 | 1.0000 | 1.000 | 0.0242 | 4 | 0 | 0 | 0 |
| act_mag | 3 | 0.502 | 0.009 | -0.00 [-0.00, 0.00] | -0.00 | 3/3 | 1.00 | 3/3 | 1.0000 | 1.000 | 0.0242 | 4 | 0 | 0 | 0 |
| grad_act | 3 | 0.500 | 0.004 | -0.01 [-0.01, -0.01] | -0.01 | 3/3 | 0.75 | 3/3 | 1.0000 | 1.000 | 0.0186 | 4 | 0 | 0 | 0 |
| add_finetune | 3 | 0.501 | 0.004 | -0.00 [-0.00, -0.00] | -0.00 | 3/3 | 0.00 | 0/3 | 1.0000 | 1.000 | 0.0303 | 0 | 4 | 0 | 0 |
| reference | 3 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 3/3 | 1.00 | 3/3 | 1.0000 | 1.000 | 0.0408 | 4 | 0 | 0 | 0 |
| gt_finetune | 3 | 0.502 | 0.009 | -0.00 [-0.00, 0.00] | -0.00 | 3/3 | 1.00 | 3/3 | 1.0000 | 1.000 | 0.0242 | 4 | 0 | 0 | 0 |
| oracle_restore | 1 | 1.000 | 1.000 | 1.00 [1.00, 1.00] | 1.00 | 1/1 | 1.00 | 1/1 | 0.9992 | 1.000 | 0.0102 | 4 | 0 | 0 | 0 |
| unconstrained | 3 | 0.501 | 0.004 | -0.00 [-0.00, -0.00] | -0.00 | 3/3 | 0.01 | 3/3 | 1.0000 | 1.000 | 0.0589 | 768 | 0 | 0 | 0 |
| null | 1 | 0.027 | 0.004 | -0.95 [-0.95, -0.95] | -0.95 | 1/1 | 0.00 | 0/1 | 0.9984 | 1.000 | 0.0000 | 0 | 0 | 0 | 0 |
| cheat_overcap | 3 | 0.000 | 0.000 | -1.01 [-1.01, -1.01] | 0.00 | 0/3 | 0.00 | 0/3 | nan | nan | nan | nan | nan | 3 | 3 |
| cheat_tamper | 3 | 0.500 | 0.004 | -0.01 [-0.01, -0.01] | 0.00 | 0/3 | 0.75 | 3/3 | 1.0000 | 1.000 | 0.0186 | 4 | 0 | 0 | 3 |
| blind_expert | 3 | 1.000 | 0.999 | 1.00 [1.00, 1.00] | 1.00 | 3/3 | 0.75 | 3/3 | 1.0000 | 1.000 | 0.0334 | 4 | 0 | 0 | 0 |
| blind_zero_retrain | 3 | 0.597 | 0.194 | 0.19 [0.19, 0.19] | 0.00 | 0/3 | 0.75 | 3/3 | 0.9011 | 0.000 | 1.2666 | 4 | 0 | 0 | 0 |
| synth_data | 3 | 0.502 | 0.004 | -0.00 [-0.00, -0.00] | -0.00 | 3/3 | 0.75 | 3/3 | 1.0000 | 1.000 | 0.0190 | 4 | 0 | 0 | 0 |

`oracle_restore` resets the planted neurons to their pre-defect values using sealed data (a ceiling, not an agent). `gt_finetune` is the blind edit procedure on the true neurons. `unconstrained` edits every neuron (k check bypassed) and is reported only as the cost of locality. `add_finetune` is the additive baseline: 4 new neurons in random layers, trained on dev with the fixed procedure. `blind_expert` is a scripted expert that reads only the workspace (probe every last digit, attribute the wrong class, keep neurons whose ablation breaks dev rows, delete the rest, retrain). `blind_zero_retrain` deletes the 4 attributed neurons and retrains them on dev. `synth_data` fine-tunes the attributed neurons on unlimited inputs labelled by the rule dev reveals (every last digit except 8). Rejected runs score y = 0. Tampered runs have gated s = 0.
