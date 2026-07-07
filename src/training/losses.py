"""Multi-task loss for DAVID-Net. See docs/02_architecture.md §8."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    v: float = 1.0
    a: float = 1.0
    quad: float = 0.5
    loc: float = 0.5
    sync: float = 0.3
    disentangle: float = 0.1


def focal_bce(logits, targets, gamma: float = 2.0, pos_weight=None):
    """Focal binary cross-entropy for class-imbalanced authenticity heads."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * ce).mean()


def disentangle_loss(z_v, z_a, z_c):
    """Orthogonality penalty pushing consistency features off the authenticity subspaces."""
    if z_c.numel() == 0:
        return z_v.new_zeros(())
    # project z_c to z_v/z_a width by truncation/mean for a cheap cosine proxy
    d = z_v.size(-1)
    zc = z_c[..., :d] if z_c.size(-1) >= d else F.pad(z_c, (0, d - z_c.size(-1)))
    cos_v = F.cosine_similarity(z_v, zc, dim=-1).abs().mean()
    cos_a = F.cosine_similarity(z_a, zc, dim=-1).abs().mean()
    return cos_v + cos_a


def localization_loss(loc_logits, seg_targets):
    """Per-frame BCE against manipulated-interval masks. seg_targets: (B, L) in {0,1}."""
    if seg_targets is None:
        return loc_logits.new_zeros(())
    L = loc_logits.size(1)
    if seg_targets.size(1) != L:
        seg_targets = F.interpolate(seg_targets.unsqueeze(1).float(), size=L, mode="nearest").squeeze(1)
    return F.binary_cross_entropy_with_logits(loc_logits, seg_targets.float())


def supcon_loss(features, labels, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al. 2020).

    features: (B, d) — will be L2-normalized here.
    labels:   (B,)   — samples sharing a label are positives for each other.
    Used by QACP with a *different* label partition per embedding space
    (docs/02_architecture.md §8b): z_v ~ video-auth, z_a ~ audio-auth, z_c ~ sync-state.
    """
    f = F.normalize(features, dim=-1)
    sim = f @ f.t() / temperature                          # (B, B)
    B = f.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=f.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye

    # log-softmax over all non-self pairs
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    n_pos = pos_mask.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return features.sum() * 0.0  # grad-connected zero (e.g. batch has no positive pairs)
    # zero-out non-positives BEFORE summing (log_prob has -inf on the diagonal; -inf*0=NaN)
    mean_log_prob_pos = log_prob.masked_fill(~pos_mask, 0.0).sum(1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


def qacp_loss(out: dict, batch: dict, temperature: float = 0.1):
    """Factorized SupCon over the three embedding spaces (QACP Stage 0)."""
    l_v = supcon_loss(out["z_v"], batch["video_label"], temperature)
    l_a = supcon_loss(out["z_a"], batch["audio_label"], temperature)
    l_c = supcon_loss(out["z_c"], batch["sync_label"], temperature) \
        if out["z_c"].numel() else out["z_v"].new_zeros(())
    total = l_v + l_a + l_c
    return total, {"qacp_v": float(l_v.detach()), "qacp_a": float(l_a.detach()),
                   "qacp_c": float(l_c.detach()), "total": float(total.detach())}


def total_loss(out: dict, batch: dict, w: LossWeights, model=None):
    v_t = batch["video_label"].float()
    a_t = batch["audio_label"].float()
    l_v = focal_bce(out["logit_v"], v_t)
    l_a = focal_bce(out["logit_a"], a_t)
    l_quad = F.cross_entropy(out["logit_quad"], batch["quadrant"].long())
    l_loc = localization_loss(out["loc_v"], batch.get("video_seg_mask")) \
        + localization_loss(out["loc_a"], batch.get("audio_seg_mask"))
    l_dis = disentangle_loss(out["z_v"], out["z_a"], out["z_c"])

    l_sync = out["logit_v"].new_zeros(())
    if model is not None and getattr(model, "sync", None) is not None and out["sync_pack"] is not None:
        vv, aa = out["sync_pack"]
        # only enforce sync on genuinely-synced (real-real) samples
        rr = (v_t == 0) & (a_t == 0)
        if rr.any():
            l_sync = model.sync.contrastive_loss(vv[rr], aa[rr])

    total = (w.v * l_v + w.a * l_a + w.quad * l_quad + w.loc * l_loc
             + w.sync * l_sync + w.disentangle * l_dis)
    parts = {"v": l_v, "a": l_a, "quad": l_quad, "loc": l_loc, "sync": l_sync, "dis": l_dis, "total": total}
    return total, {k: float(v.detach()) for k, v in parts.items()}
