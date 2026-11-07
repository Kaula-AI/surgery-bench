"""Dev-set accuracy of model.pt, with submission.npz applied if it exists. A convenience tool: edit it freely."""
import os

import numpy as np
import torch

from model import add_neurons, load_model, set_neuron

HERE = os.path.dirname(os.path.abspath(__file__))
m = load_model(os.path.join(HERE, "model.pt"))
sub_path = os.path.join(HERE, "submission.npz")
if os.path.exists(sub_path):
    s = np.load(sub_path)
    if "layer" in s.files:
        for j in range(len(s["layer"])):
            set_neuron(m, int(s["layer"][j]), int(s["index"][j]), s["w_in"][j], s["b_in"][j], s["w_out"][j])
    if "add_layer" in s.files:
        for j in range(len(s["add_layer"])):
            add_neurons(m, int(s["add_layer"][j]), s["add_w_in"][j:j + 1], s["add_b_in"][j:j + 1], s["add_w_out"][j][:, None])
d = np.load(os.path.join(HERE, "dev.npz"))
with torch.no_grad():
    pred = m(torch.as_tensor(d["x"])).argmax(-1).numpy()
print(f"dev accuracy {np.mean(pred == d['y']):.4f} ({'with' if os.path.exists(sub_path) else 'without'} submission)")
