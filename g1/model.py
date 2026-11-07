"""Tiny transformer with addressable MLP neurons.

This file is copied into the agent's workspace, so it is the agent's documentation of the architecture.

Neuron (layer l, index i) is the triple
    blocks[l].mlp_in.weight[i, :]   input row   (d_model,)
    blocks[l].mlp_in.bias[i]        input bias  ()
    blocks[l].mlp_out.weight[:, i]  output col  (d_model,)
Classification reads the residual stream at the last position.

Neurons can also be added: add_neurons(model, l, w_in, b_in, w_out) appends new neurons to layer l's MLP, at indices
d_mlp, d_mlp + 1, ... of that layer. load_model reads checkpoints whose layers have grown.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(vocab=10, seq_len=8, n_classes=10, d_model=64, n_heads=4, d_mlp=256, n_layers=3)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)

    def forward(self, x, ablate=None, acts=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        a = F.gelu(self.mlp_in(self.ln2(x)))
        if ablate:
            a = a.clone()
            a[..., ablate] = 0.0
        if acts is not None:
            acts.append(a)
        return x + self.mlp_out(a)


class TinyTransformer(nn.Module):
    def __init__(self, vocab, seq_len, n_classes, d_model, n_heads, d_mlp, n_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_mlp) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, ablate=None, acts=None):
        """ablate: {layer: [neuron indices]} zeroed after the GELU. acts: list that collects each layer's activations."""
        h = self.tok(x) + self.pos
        for l, b in enumerate(self.blocks):
            h = b(h, (ablate or {}).get(l), acts)
        return self.head(self.ln_f(h[:, -1]))


def new_model():
    return TinyTransformer(**CONFIG)


def load_model(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = new_model()
    for l, b in enumerate(m.blocks):  # a grown layer has more MLP neurons than CONFIG says
        extra = sd[f"blocks.{l}.mlp_in.weight"].shape[0] - b.mlp_in.out_features
        if extra > 0:
            add_neurons(m, l, torch.zeros(extra, b.mlp_in.in_features), torch.zeros(extra),
                        torch.zeros(b.mlp_out.out_features, extra))
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def add_neurons(model, l, w_in, b_in, w_out):
    """Append k new neurons to layer l's MLP. w_in (k, d_model), b_in (k,), w_out (d_model, k). Returns their indices."""
    b = model.blocks[l]
    w_in, b_in, w_out = (torch.as_tensor(a, dtype=torch.float32) for a in (w_in, b_in, w_out))
    n, k = b.mlp_in.out_features, w_in.shape[0]
    new_in, new_out = nn.Linear(b.mlp_in.in_features, n + k), nn.Linear(n + k, b.mlp_out.out_features)
    new_in.weight.copy_(torch.cat([b.mlp_in.weight, w_in]))
    new_in.bias.copy_(torch.cat([b.mlp_in.bias, b_in]))
    new_out.weight.copy_(torch.cat([b.mlp_out.weight, w_out], 1))
    new_out.bias.copy_(b.mlp_out.bias)
    b.mlp_in, b.mlp_out = new_in, new_out
    return list(range(n, n + k))


def get_neuron(model, l, i):
    b = model.blocks[l]
    return (b.mlp_in.weight[i].detach().clone(), b.mlp_in.bias[i].detach().clone(),
            b.mlp_out.weight[:, i].detach().clone())


@torch.no_grad()
def set_neuron(model, l, i, w_in, b_in, w_out):
    b = model.blocks[l]
    b.mlp_in.weight[i] = torch.as_tensor(w_in, dtype=torch.float32)
    b.mlp_in.bias[i] = torch.as_tensor(b_in, dtype=torch.float32)
    b.mlp_out.weight[:, i] = torch.as_tensor(w_out, dtype=torch.float32)
