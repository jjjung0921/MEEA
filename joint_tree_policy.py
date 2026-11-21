"""One-reaction tree SSM for ranking a fixed shortlist of templates."""

import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import rdChemReactions


def template_tree_features(rules, width=128):
    def fingerprint(mol):
        return np.asarray(Chem.PatternFingerprint(mol, fpSize=width), dtype=bool)

    roots, children = [], []
    for rule in rules:
        reaction = rdChemReactions.ReactionFromSmarts(rule)
        root = np.logical_or.reduce([fingerprint(mol) for mol in reaction.GetReactants()])
        branches = [fingerprint(fragment) for mol in reaction.GetProducts()
                    for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)]
        roots.append(root)
        children.append(branches)
    features = np.zeros((len(rules), max(map(len, children)) + 1, width), dtype=bool)
    mask = np.zeros(features.shape[:2], dtype=bool)
    for i, (root, branches) in enumerate(zip(roots, children)):
        features[i, :len(branches) + 1] = [root, *branches]
        mask[i, :len(branches) + 1] = True
    return features, mask


class TemplateTreeSSM(nn.Module):
    """Input-conditioned diagonal tree transition and masked AND-sibling messages."""

    def __init__(self, features, mask, joint, width=48):
        super().__init__()
        self.register_buffer("features", torch.as_tensor(features), persistent=False)
        self.register_buffer("mask", torch.as_tensor(mask), persistent=False)
        self.joint = joint
        self.target = nn.Linear(2048, width)
        self.pattern = nn.Linear(features.shape[-1], width)
        self.transition = nn.Linear(width, 2 * width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.message = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(2 * width, width)
        self.readout = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.scale = width ** -0.5

    def forward(self, fp, candidate_ids, prior):
        parts = self.pattern(self.features[candidate_ids].to(fp.dtype))
        valid = self.mask[candidate_ids][..., 1:]
        root = torch.tanh(self.target(fp)[:, None] + parts[:, :, 0])
        decay, drive = self.transition(parts[:, :, 1:]).chunk(2, dim=-1)
        child = decay.sigmoid() * root[:, :, None] + drive.tanh()
        if self.joint:
            scores = (self.query(child) @ self.key(child).transpose(-2, -1)) * self.scale
            diagonal = torch.eye(valid.shape[-1], dtype=torch.bool, device=valid.device)
            allowed = valid.unsqueeze(-1) & valid.unsqueeze(-2) & ~diagonal
            weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(dim=-1)
            message = (weights * allowed) @ self.message(child)
            child = child + self.gate(torch.cat((child, message), dim=-1)).sigmoid() * message
        pooled = (child * valid.unsqueeze(-1)).sum(dim=-2) / valid.sum(dim=-1, keepdim=True).clamp_min(1)
        return prior + self.readout(torch.cat((root, pooled, root * pooled), dim=-1)).squeeze(-1)


class ShortlistPolicy(nn.Module):
    def __init__(self, proposal, ranker, shortlist=50):
        super().__init__()
        self.proposal = proposal
        self.ranker = ranker
        self.shortlist = shortlist

    def forward(self, fingerprint):
        base = self.proposal(fingerprint)
        prior, candidates = base.topk(self.shortlist, dim=-1)
        reranked = self.ranker(fingerprint, candidates, prior)
        # Preserve the proposal set and normalize over the same template space.
        return base.masked_fill(torch.ones_like(base, dtype=torch.bool).scatter_(1, candidates, False),
                                -torch.inf).scatter(1, candidates, reranked)
