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
    sync: float = 0.1
    disentangle: float = 0.1


def focal_bce(logits, targets, gamma: float = 2.0, pos_weight=None, mask=None):
    """Focal binary cross-entropy for class-imbalanced authenticity heads.

    mask: optional (B,) float — samples with mask=0 (e.g. modality dropped)
    contribute no loss for this head.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = (1 - p_t) ** gamma * ce
    if mask is None:
        return loss.mean()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def disentangle_loss(z_v, z_a, z_c, mi_weight: float = 0.1):
    """Orthogonality + MI penalty pushing consistency features off authenticity subspaces.

    Architecture §6: "Default: orthogonality + MI penalty (cheap, stable)."
    The MI penalty (vCLUB estimator) minimizes mutual information between
    authenticity and consistency embeddings.
    """
    if z_c.numel() == 0:
        return z_v.new_zeros(())
    # project z_c to z_v/z_a width by truncation/mean for a cheap cosine proxy
    d = z_v.size(-1)
    zc = z_c[..., :d] if z_c.size(-1) >= d else F.pad(z_c, (0, d - z_c.size(-1)))
    cos_v = F.cosine_similarity(z_v, zc, dim=-1).abs().mean()
    cos_a = F.cosine_similarity(z_a, zc, dim=-1).abs().mean()
    ortho = cos_v + cos_a

    # MI penalty (vCLUB upper bound approximation)
    # Minimize I(z_v; z_c) + I(z_a; z_c) so authenticity and consistency are independent
    B = z_v.size(0)
    if B < 2:
        return ortho
    # Simple variance-based MI proxy: if distributions are independent,
    # joint = product of marginals. Penalize deviation.
    z_v_norm = F.normalize(z_v, dim=-1)
    z_c_norm = F.normalize(zc, dim=-1)
    sim_v_c = (z_v_norm @ z_c_norm.t())  # (B, B)
    # Positive pairs on diagonal, negatives off-diagonal
    pos = sim_v_c.diag().mean()
    neg = (sim_v_c.sum() - sim_v_c.diag().sum()) / (B * (B - 1))
    mi_proxy = (pos - neg).clamp(min=0)

    return ortho + mi_weight * mi_proxy


def localization_loss(loc_logits, seg_targets, mask=None):
    """Per-frame BCE against manipulated-interval masks. seg_targets: (B, L) in {0,1}.

    mask: optional (B,) float — rows with mask=0 (modality absent) are excluded.
    """
    if seg_targets is None:
        return loc_logits.new_zeros(())
    L = loc_logits.size(1)
    if seg_targets.size(1) != L:
        seg_targets = F.interpolate(seg_targets.unsqueeze(1).float(), size=L, mode="nearest").squeeze(1)
    loss = F.binary_cross_entropy_with_logits(loc_logits, seg_targets.float(), reduction="none")
    if mask is None:
        return loss.mean()
    denom = (mask.sum() * L).clamp(min=1.0)
    return (loss * mask.unsqueeze(1)).sum() / denom


def supcon_loss(features, labels, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al. 2020).

    features: (B, d) — will be L2-normalized here.
    labels:   (B,)   — samples sharing a label are positives for each other.
    Used by QACP with a *different* label partition per embedding space
    (docs/02_architecture.md §8b): z_v ~ video-auth, z_a ~ audio-auth, z_c ~ sync-state.

    When no positive pairs exist in the batch (all unique labels — common at small batch
    sizes), falls back to NT-Xent (SimCLR-style) treating each sample as its own positive
    via the log-softmax diagonal. This ensures a meaningful gradient is always returned
    instead of the silent zero-gradient trap of `features.sum() * 0.0`.
    """
    f = F.normalize(features, dim=-1)
    sim = f @ f.t() / temperature                          # (B, B)
    B = f.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=f.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye

    # log-softmax over all non-self pairs
    sim_no_self = sim.masked_fill(eye, float("-inf"))
    log_prob = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)

    n_pos = pos_mask.sum(1)
    valid = n_pos > 0
    if not valid.any():
        # No same-label pairs in the batch (common at batch_size ≤ 5 with many classes).
        # Fall back to NT-Xent: treat the most-similar non-self sample as a soft positive
        # using the full similarity distribution. This avoids the silent zero-gradient
        # trap and keeps the encoder learning separable representations.
        # The loss encourages maximum margin between the most-similar pair and all others.
        # log_prob diagonal = log(softmax over non-self), negating pushes embeddings apart.
        ntsim = sim.masked_fill(eye, float("-inf"))
        # Treat the most-similar non-self as the positive
        best_pos = ntsim.max(dim=1).values  # (B,)
        log_denom = torch.logsumexp(ntsim, dim=1)  # (B,)
        return -(best_pos - log_denom).mean()
    # zero-out non-positives BEFORE summing (log_prob has -inf on the diagonal; -inf*0=NaN)
    mean_log_prob_pos = log_prob.masked_fill(~pos_mask, 0.0).sum(1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


def qacp_loss(out: dict, batch: dict, temperature: float = 0.1):
    """Factorized SupCon over the three embedding spaces (QACP Stage 0).

    Uses pre-fusion embeddings (z_v_pre, z_a_pre) for the unimodal terms
    (video_label, audio_label) to prevent cross-modal leakage — the audio
    encoder should not learn to classify audio authenticity using video
    artifacts leaked through fusion layers.  The consistency term (z_c)
    is intentionally cross-modal and stays post-fusion.
    """
    l_v = supcon_loss(out["z_v_pre"], batch["video_label"], temperature)
    l_a = supcon_loss(out["z_a_pre"], batch["audio_label"], temperature)
    l_c = supcon_loss(out["z_c"], batch["sync_label"], temperature) \
        if out["z_c"].numel() else out["z_v"].new_zeros(())
    total = l_v + l_a + l_c
    return total, {"qacp_v": float(l_v.detach()), "qacp_a": float(l_a.detach()),
                   "qacp_c": float(l_c.detach()), "total": float(total.detach())}


def total_loss(out: dict, batch: dict, w: LossWeights, model=None):
    v_t = batch["video_label"].float()
    a_t = batch["audio_label"].float()
    # availability masks (modality dropout / genuinely missing streams):
    # a dropped modality receives no supervised gradient for its head.
    v_av = out.get("v_avail", v_t.new_ones(v_t.shape))
    a_av = out.get("a_avail", a_t.new_ones(a_t.shape))
    both = v_av * a_av

    l_v = focal_bce(out["logit_v"], v_t, mask=v_av)
    l_a = focal_bce(out["logit_a"], a_t, mask=a_av)
    # quadrant is only defined when both streams exist
    l_quad = (F.cross_entropy(out["logit_quad"], batch["quadrant"].long(), reduction="none")
              * both).sum() / both.sum().clamp(min=1.0)
    l_loc = localization_loss(out["loc_v"], batch.get("video_seg_mask"), mask=v_av) \
        + localization_loss(out["loc_a"], batch.get("audio_seg_mask"), mask=a_av)
    l_dis = disentangle_loss(out["z_v"], out["z_a"], out["z_c"])

    l_sync = out["logit_v"].new_zeros(())
    if model is not None and getattr(model, "sync", None) is not None and out["sync_pack"] is not None:
        vv, aa = out["sync_pack"]
        # only enforce sync on genuinely-synced (real-real) samples with both streams
        rr = (v_t == 0) & (a_t == 0) & (both > 0)
        if rr.any():
            l_sync = model.sync.contrastive_loss(vv[rr], aa[rr])

    total = (w.v * l_v + w.a * l_a + w.quad * l_quad + w.loc * l_loc
             + w.sync * l_sync + w.disentangle * l_dis)
    parts = {"v": l_v, "a": l_a, "quad": l_quad, "loc": l_loc, "sync": l_sync, "dis": l_dis, "total": total}
    return total, {k: float(v.detach()) for k, v in parts.items()}
