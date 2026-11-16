#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared AGI Core v8.1.0: QK-Norm HyperGraph, FROG Global-Max Context, 3-Head Ensemble WM."""
from __future__ import annotations
import math, os, random
from contextlib import nullcontext
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


def amp_ctx(device):
    return torch.amp.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()


def host_mib():
    if resource is None:
        return 0.0
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if os.name != "posix" or "darwin" not in os.sys.platform else r / 2**20


def lr_at(frac, warm=0.03):
    if frac < warm:
        return frac / warm
    t = (frac - warm) / max(1e-9, 1 - warm)
    return 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * t))


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_bce(a, b):
    if a.numel() == 0:
        return a.sum() * 0.0
    return F.binary_cross_entropy_with_logits(a, b)


def masked_logsoftmax(logits, mask):
    neg = torch.finfo(logits.dtype).min / 2
    return torch.log_softmax(logits.masked_fill(~mask, neg), dim=-1)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


def ece(conf, correct, bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


class SelfAttnSet(nn.Module):
    def __init__(self, C, heads, qk_norm=True):
        super().__init__()
        self.h = heads; self.dk = C // heads; self.qk_norm = qk_norm
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        if qk_norm:
            self.qk_scale = nn.Parameter(torch.full((1, heads, 1, 1), 3.0))

    def _attn(self, x):
        N, S, C = x.shape
        qkv = self.qkv(x).reshape(N, S, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.qk_scale.clamp(0.5, 6.0)
            k = F.normalize(k, dim=-1)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(N, S, C))

    def forward(self, x):
        N = x.shape[0]
        if N <= 32768:
            return self._attn(x)
        return torch.cat([self._attn(x[i:i + 32768]) for i in range(0, N, 32768)], dim=0)


class HyperGraphBlock(nn.Module):
    def __init__(self, C, heads, attn_names, qk_norm=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(C, eps=1e-5)
        self.group_attns = nn.ModuleDict({g: SelfAttnSet(C, heads, qk_norm=qk_norm) for g in attn_names})
        self.ln2 = nn.LayerNorm(C, eps=1e-5)
        self.attn_slot = SelfAttnSet(C, heads, qk_norm=qk_norm)
        self.ln3 = nn.LayerNorm(C, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(C, 3 * C), nn.GELU(), nn.Linear(3 * C, C))

    def forward(self, h, unit_specs):
        B, N, K, C = h.shape
        u = self.ln1(h)
        msg = 0
        for attn_key, weight, u_idx, inv_idx, deg in unit_specs:
            M, S = u_idx.shape
            sel = u[:, u_idx]
            x_g = sel.permute(0, 1, 3, 2, 4).reshape(B * M * K, S, C)
            o_g = self.group_attns[attn_key](x_g)
            o_flat = o_g.reshape(B, M, K, S, C).permute(0, 1, 3, 2, 4).reshape(B, M * S, K, C)
            if inv_idx is not None:
                acc = o_flat[:, inv_idx]
            else:
                acc = torch.zeros(B, N, K, C, dtype=o_flat.dtype, device=o_flat.device)
                acc.index_add_(1, u_idx.reshape(-1), o_flat)
                acc = acc / deg
            msg = msg + weight * acc.to(h.dtype)
        h = h + msg
        if K > 1:
            us = self.ln2(h).reshape(B * N, K, C)
            h = h + self.attn_slot(us).reshape(B, N, K, C)
        h = h + self.mlp(self.ln3(h))
        return h


class RelationalReasoner(nn.Module):
    def __init__(self, n_nodes, unit_groups_cfg, feat_tie_idx, C=96, heads=4, L=3, T_max=24, T_cyc=8, qk_norm=True):
        super().__init__()
        self.N, self.C, self.L, self.T_max, self.T_cyc = n_nodes, C, L, T_max, T_cyc
        f_uniq = max(feat_tie_idx) + 1
        self.register_buffer("fidx", torch.as_tensor(feat_tie_idx, dtype=torch.long))
        self.unit_meta = []
        attn_names = []
        for name, attn_key, weight, arr in unit_groups_cfg:
            if attn_key not in attn_names:
                attn_names.append(attn_key)
            t_idx = torch.as_tensor(arr, dtype=torch.long)
            flat = t_idx.reshape(-1)
            deg = torch.bincount(flat, minlength=n_nodes).float().clamp(min=1.0)
            is_part = (flat.numel() == n_nodes) and bool((torch.bincount(flat, minlength=n_nodes) == 1).all())
            self.register_buffer(f"ug_{name}", t_idx)
            self.register_buffer(f"ud_{name}", deg.view(1, n_nodes, 1, 1))
            if is_part:
                self.register_buffer(f"inv_{name}", torch.argsort(flat))
            self.unit_meta.append((name, attn_key, weight, is_part))

        self.Win = nn.Parameter(torch.randn(f_uniq, C) * (1.0 / math.sqrt(f_uniq)))
        self.bin = nn.Parameter(torch.zeros(C))
        self.inp = nn.Linear(C, C)
        self.step_gate = nn.Parameter(torch.full((C,), 0.5))
        self.y_proj = nn.Linear(1, C, bias=False)
        self.step_emb = nn.Parameter(torch.zeros(T_max, C))
        self.blocks = nn.ModuleList([HyperGraphBlock(C, heads, attn_names, qk_norm=qk_norm) for _ in range(L)])
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        self.head = nn.Linear(C, 1)
        self.cand = nn.Linear(C, 1)
        self.lnc = nn.LayerNorm(C, eps=1e-5)
        # FROG v3.2: [mean, amax] global context prevents 1/N dilution of single-cell conflicts!
        self.pool_proj = nn.Sequential(nn.Linear(2 * C, C), nn.LayerNorm(C, eps=1e-5))
        self.value = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))
        self.conf = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))
        self.dead_head = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))

        # FROG v3.2: Equivariant 3-Head Ensemble Dynamics + Action-Centered Conflict Head
        self.act_emb = nn.Linear(1, C, bias=False)
        self.dyn_blk = HyperGraphBlock(C, heads, attn_names, qk_norm=qk_norm)
        self.dyn_deltas = nn.ModuleList([nn.Linear(C, C) for _ in range(3)])
        self.dyn_norm = nn.LayerNorm(C, eps=1e-5)
        self.coll_head = nn.Linear(C, 1)

    def _unit_specs(self):
        return [(ak, w, getattr(self, f"ug_{nm}"),
                 (getattr(self, f"inv_{nm}") if isp else None),
                 getattr(self, f"ud_{nm}"))
                for nm, ak, w, isp in self.unit_meta]

    def _global_ctx(self, z):
        c = self.lnc(z.mean(2))  # (B, N, C)
        return self.pool_proj(torch.cat([c.mean(1), c.amax(1)], dim=-1))

    def _predict_from_z(self, z):
        gsum = self._global_ctx(z)
        return (
            self.head(z).squeeze(-1).float(),
            self.cand(z).squeeze(-1).float(),
            self.value(gsum).squeeze(-1).float(),
            self.conf(gsum).squeeze(-1).float(),
            self.dead_head(gsum).squeeze(-1).float(),
        )

    def forward_features(self, f, legal_bool=None, T=6, ckpt=False, all_steps=False, state=None,
                         step_off=0, noise_std=0.0, focus_temp=1.0):
        unit_specs = self._unit_specs()
        W = self.Win[self.fidx]
        h0 = self.inp(F.gelu(torch.einsum("bnkf,fc->bnkc", f, W) + self.bin))
        if state is None:
            h, lg_prev = h0, None
        else:
            h, lg_prev = state
        if noise_std > 0.0:
            h_std = h.detach().std(dim=(-2, -1), keepdim=True).clamp(min=0.5)
            h = h + noise_std * h_std * torch.randn_like(h)
        alpha = torch.sigmoid(self.step_gate)
        if legal_bool is not None:
            has_leg = legal_bool.any(-1, keepdim=True).float()
        outs = []
        for t in range(T):
            ti = (step_off + t) % self.T_cyc
            if t > 0 or state is not None:
                y_fb = 0
                if lg_prev is not None:
                    scaled = lg_prev / max(focus_temp, 0.25)
                    if legal_bool is not None:
                        y_t = torch.softmax(scaled.masked_fill(~legal_bool, -1e4), dim=-1) * has_leg
                    else:
                        y_t = torch.softmax(scaled, dim=-1)
                    y_fb = self.y_proj(y_t.to(h.dtype).unsqueeze(-1))
                h = (1.0 - alpha) * h + alpha * (h0 + y_fb) + self.step_emb[ti]
            else:
                h = h + self.step_emb[ti]
            for blk in self.blocks:
                if ckpt and torch.is_grad_enabled():
                    h = _ckpt(blk, h, unit_specs, use_reentrant=False)
                else:
                    h = blk(h, unit_specs)
            lg_prev = self.head(self.lnf(h)).squeeze(-1).float()
            if all_steps or t == T - 1:
                outs.append(lg_prev)
        z = self.lnf(h)
        _, cand_lg, val_lg, conf_lg, dead_lg = self._predict_from_z(z)
        return dict(logits=outs[-1], logits_all=torch.stack(outs),
                    cand=cand_lg, value=val_lg, conf=conf_lg, dead=dead_lg,
                    _h=h, _lg=lg_prev, _z=z)

    def imagine_step(self, z, act_mask):
        """FROG-upgraded equivariant 3-head ensemble world model + action-centered conflict head."""
        unit_specs = self._unit_specs()
        am_f = act_mask.to(z.dtype)
        z_in = z + self.act_emb(am_f.unsqueeze(-1))
        d_feat = self.dyn_blk(z_in, unit_specs) - z_in
        z_ens = torch.stack([self.dyn_norm(z + head(d_feat)) for head in self.dyn_deltas], dim=0)  # (3, B, N, K, C)
        z_next = z_ens.mean(dim=0)
        ens_var = z_ens.var(dim=0).mean(dim=(-3, -2, -1)).float()
        lg, cand, val, conf, dead = self._predict_from_z(z_next)
        # Action-centered collision/conflict logit right at the acted (cell, slot) token!
        coll = (self.coll_head(z_next).squeeze(-1) * am_f).sum(dim=(1, 2)).float()
        return dict(z=z_next, ens_var=ens_var, logits=lg, cand=cand, value=val, conf=conf, dead=dead, coll=coll)

    @torch.no_grad()
    def logit_bound(self):
        w = self.head.weight.norm().item()
        gam = self.lnf.weight.abs().max().item()
        beta = self.lnf.bias.abs().max().item()
        return math.sqrt(self.C) * (gam * math.sqrt(self.C - 1) + beta) * w + self.head.bias.abs().max().item()


@torch.no_grad()
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float(); X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
    return X.T if transposed else X


class MuonAdamW:
    def __init__(self, model, lr_muon=0.02, lr_adam=3e-4, momentum=0.95, wd=0.01, ns_steps=5):
        self.ns = ns_steps; muon_p, adam_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            hidden2d = p.ndim == 2 and min(p.shape) >= 8 and not any(
                k in name for k in ("Win", "head", "cand", "step_emb", "value", "conf", "dead_head", "coll_head", "pool_proj", "inp", "y_proj", "act_emb"))
            (muon_p if hidden2d else adam_p).append(p)
        self.muon_p, self.adam_p = muon_p, adam_p
        self.buf = {p: torch.zeros_like(p) for p in muon_p}
        self.opt = torch.optim.AdamW(adam_p, lr=lr_adam, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
        self.lr_muon, self.mom, self.wd = lr_muon, momentum, wd; self._adam_lr = lr_adam; self.stats = {}

    def zero_grad(self):
        for p in self.muon_p + self.adam_p: p.grad = None

    @torch.no_grad()
    def step(self, lr_scale=1.0):
        smax, smin, n = 0.0, 1e9, 0
        for p in self.muon_p:
            if p.grad is None: continue
            g = p.grad; buf = self.buf[p]; buf.mul_(self.mom).add_(g); u = g.add(buf, alpha=self.mom)
            o = newton_schulz5(u, self.ns); scale = 0.2 * math.sqrt(max(p.shape))
            p.mul_(1 - self.lr_muon * lr_scale * self.wd); p.add_(o, alpha=-self.lr_muon * lr_scale * scale)
            if random.random() < 0.05:
                s = torch.linalg.svdvals(o.float()); smax = max(smax, s.max().item()); smin = min(smin, s.min().item()); n += 1
        for pg in self.opt.param_groups: pg["lr"] = self._adam_lr * lr_scale
        self.opt.step()
        self.stats = dict(muon_sigma_max=(smax if n else None), muon_sigma_min=(smin if n else None), muon_sampled=n)

    def set_lrs(self, lr_muon, adam_lr): self.lr_muon, self._adam_lr = lr_muon, adam_lr


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay; self.steps = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.steps += 1; d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else: self.shadow[k].copy_(v)