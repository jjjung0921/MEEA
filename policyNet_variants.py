"""
policyNet_variants.py
=====================
Alternative policy networks that keep the same fingerprint input as policyNet:
1) DropoutPolicyNet: deeper MLP with dropout on every hidden block
2) TransformerPolicyNet: patchify fingerprints and pass through a Transformer encoder
"""

import gzip
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from policyNet import preprocess, merge, train


class DropoutPolicyNet(nn.Module):
    """Two hidden-layer MLP with dropout on every block."""

    def __init__(self, n_rules: int, fp_dim: int = 2048, hidden: int = 512, dropout_rate: float = 0.4):
        super().__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, n_rules),
        )

    def forward(self, x, y=None, loss_fn=nn.CrossEntropyLoss()):
        logits = self.net(x)
        if y is not None:
            return loss_fn(logits, y)
        return logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the Transformer encoder."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerPolicyNet(nn.Module):
    """
    Patchify fingerprint bits and encode them with a Transformer encoder.
    """

    def __init__(
        self,
        n_rules: int,
        fp_dim: int = 2048,
        patch_size: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fp_dim % patch_size != 0:
            raise ValueError(f"fp_dim ({fp_dim}) must be divisible by patch_size ({patch_size})")
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.seq_len = fp_dim // patch_size
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(patch_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_rules))

    def forward(self, x, y=None, loss_fn=nn.CrossEntropyLoss()):
        # x: (batch, fp_dim)
        bsz = x.size(0)
        x = x.view(bsz, self.seq_len, self.patch_size)
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # simple global average pooling
        logits = self.head(x)
        if y is not None:
            return loss_fn(logits, y)
        return logits


def _load_templates(template_rule_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    template_rules = {}
    with open(template_rule_path, "r") as f:
        for i, l in enumerate(f):
            rule = l.strip()
            template_rules[rule] = i
    idx2rule = {idx: rule for rule, idx in template_rules.items()}
    return template_rules, idx2rule


def load_policy_variant(
    model_type: str,
    state_path: str,
    template_rule_path: str,
    fp_dim: int = 2048,
    **model_kwargs,
):
    template_rules, idx2rule = _load_templates(template_rule_path)
    n_rules = len(template_rules)

    if model_type == "dropout":
        net = DropoutPolicyNet(n_rules=n_rules, fp_dim=fp_dim, **model_kwargs)
    elif model_type == "transformer":
        net = TransformerPolicyNet(n_rules=n_rules, fp_dim=fp_dim, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint = torch.load(state_path, map_location="cpu")
    # DataParallel로 학습된 경우 키 앞의 'module.' 접두사를 제거
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    net.load_state_dict(checkpoint)
    return net, idx2rule


@dataclass
class PolicyVariantModel:
    """
    Inference wrapper that mirrors MLPModel but works with policy variants.
    """

    model_type: str
    state_path: str
    template_path: str
    device: str = "cpu"
    fp_dim: int = 2048
    model_kwargs: dict = None

    def __post_init__(self):
        kwargs = self.model_kwargs or {}
        self.net, self.idx2rules = load_policy_variant(
            self.model_type, self.state_path, self.template_path, self.fp_dim, **kwargs
        )
        self.net.eval()
        self.net.to(self.device)

    def run(self, x: str, topk: int = 10):
        arr = preprocess(x, self.fp_dim)
        arr = torch.tensor(arr.reshape(1, -1), dtype=torch.float32, device=self.device)
        preds = self.net(arr)
        preds = F.softmax(preds, dim=1).cpu()

        probs, idx = torch.topk(preds, k=topk)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]

        reactants, scores, templates = [], [], []
        for i, rule in enumerate(rule_k):
            # rdchiral import is heavy; import lazily
            from rdchiral.main import rdchiralRunText

            try:
                out1 = rdchiralRunText(rule, x)
                if not out1:
                    continue
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item() / len(out1))
                    templates.append(rule)
            except (ValueError, RuntimeError, KeyError, IndexError):
                continue

        if not reactants:
            return None

        reactants_d = {}
        for r, s, t in zip(reactants, scores, templates):
            if "." in r:
                norm_r = ".".join(sorted(r.strip().split(".")))
            else:
                norm_r = r
            reactants_d.setdefault(norm_r, []).append((s, t))

        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]

        return {"reactants": reactants, "scores": scores, "template": templates}


def train_policy_variant(
    model_type: str,
    batch_size: int = 1024,
    lr: float = 0.001,
    epochs: int = 100,
    weight_decay: float = 0,
    dropout_rate: float = 0.4,
    saved_model: str = "./model/saved_variant_state",
    **model_kwargs,
):
    """
    Train a policy variant using the same inputs/pickles as policyNet.
    """
    with gzip.open("./prepare_data/uspto_template.pkl.gz", "rb") as f:
        templates = pickle.load(f)
    num_of_rules = len(templates)

    if model_type == "dropout":
        net = DropoutPolicyNet(n_rules=num_of_rules, dropout_rate=dropout_rate, **model_kwargs)
    elif model_type == "transformer":
        net = TransformerPolicyNet(n_rules=num_of_rules, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with gzip.open("./prepare_data/policyTrain.pkl.gz", "rb") as f:
        train_data = pickle.load(f)
    with gzip.open("./prepare_data/policyTest.pkl.gz", "rb") as f:
        test_data = pickle.load(f)

    train(
        net,
        dataTrain=train_data,
        dataTest=test_data,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        wd=weight_decay,
        saved_model=saved_model,
    )


__all__ = [
    "DropoutPolicyNet",
    "TransformerPolicyNet",
    "PolicyVariantModel",
    "train_policy_variant",
    "load_policy_variant",
]
