"""Why QACP Stage 0 stops learning: the SupCon terms saturate at their loss floor.

Observed on Kaggle (both runs, every epoch past ~5):

    qacp_v=-0.0000  qacp_a=0.6932  qacp_c=1.0986  gnorm=0.00

Those are analytic constants, not convergence.

The mechanism -- NOT representation collapse; the embeddings stay well spread
----------------------------------------------------------------------------
`supcon_loss` returns `-mean_i log p_i` over the non-self entries of the softmax row.
At batch_size 4 with binary labels that objective is trivially satisfiable, and each
label regime has an exact floor the optimiser reaches within a few hundred steps, after
which the gradient is zero and nothing further is learned:

    zero negatives (all 4 share a label)  -> floor ln(B-1) = 1.0986   matches qacp_c
    exactly one negative (3 pos + 1)      -> floor ln(2)   = 0.6932   matches qacp_a
    balanced 2 vs 2                       -> floor ~0                 matches qacp_v

All three observed values are reproduced below from free embedding vectors, with no
encoder and no data -- so the objective, not the model, is what is stuck.

Of the five pseudo-classes only MISMATCH is SYNC_MISMATCHED, so at batch_size 4 about
20% of batches hand the sync axis no negative at all and most of the rest hand it one.
That is why qacp_c never left 1.0986 in any run.

    python scripts/diagnose_qacp_collapse.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_quadrants import QACP_CLASSES  # noqa: E402
from src.training.losses import supcon_loss  # noqa: E402

# label table straight out of build_pseudo_sample()
LABELS = {
    "RVRA":     (0, 0, 0),
    "RVFA":     (0, 1, 0),
    "FVRA":     (1, 0, 0),
    "FVFA":     (1, 1, 0),
    "MISMATCH": (0, 0, 1),
}
AXES = ("video", "audio", "sync")


def collapse_metric(emb: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity. 1.0 = every sample on the same point."""
    f = torch.nn.functional.normalize(emb.float(), dim=-1)
    sim = f @ f.t()
    n = f.size(0)
    return float(sim[~torch.eye(n, dtype=torch.bool)].mean())


def optimise(label_fn, B=4, steps=600, dim=128, lr=1e-2, seed=0):
    """Optimise free embeddings against the real supcon_loss. No encoder, no data: if
    this collapses, the objective itself rewards collapse."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.Parameter(torch.randn(B, dim) * 0.5)
    opt = torch.optim.Adam([emb], lr=lr)
    loss = gnorm = 0.0
    for _ in range(steps):
        lab = label_fn(B, g)
        l = supcon_loss(emb, lab, temperature=0.1)
        opt.zero_grad()
        l.backward()
        gnorm = float(emb.grad.norm())
        opt.step()
        loss = float(l.detach())
    return loss, gnorm, collapse_metric(emb.detach())


def lab_zero_neg(B, g):
    return torch.zeros(B, dtype=torch.long)                       # all same label


def lab_one_neg(B, g):
    l = torch.zeros(B, dtype=torch.long)
    l[0] = 1
    return l                                                      # 3 positives, 1 negative


def lab_balanced(B, g):
    return (torch.arange(B) % 2).long()                           # 50/50


def qacp_sync_labels(B, g):
    """The real thing: B distinct pseudo-classes out of 5, sync axis."""
    idx = torch.randperm(len(QACP_CLASSES), generator=g).tolist()
    classes = [QACP_CLASSES[idx[i % len(idx)]] for i in range(B)]
    return torch.tensor([LABELS[c][2] for c in classes])


def main():
    B = 4
    print(f"batch_size = {B}   ln(B-1) = {math.log(B-1):.4f}   "
          f"<- the value qacp_c held at every step on Kaggle\n")
    print(f"{'label regime':>28} {'final loss':>11} {'gnorm':>9} {'collapse':>9}")
    print("-" * 60)
    for name, fn in (("zero negatives (all same)", lab_zero_neg),
                     ("one negative (3 pos + 1)", lab_one_neg),
                     ("balanced 50/50", lab_balanced),
                     ("REAL QACP sync axis", qacp_sync_labels)):
        loss, gnorm, col = optimise(fn, B=B)
        flag = "  <-- COLLAPSED, at exactly ln(B-1)" if col > 0.99 else ""
        print(f"{name:>28} {loss:>11.4f} {gnorm:>9.4f} {col:>9.3f}{flag}")

    print("\nHow often does the real sampler hand the sync axis zero negatives?")
    g = torch.Generator().manual_seed(1)
    for bs in (4, 8, 16, 32):
        zero = sum(1 for _ in range(4000) if len(qacp_sync_labels(bs, g).unique()) == 1)
        print(f"  batch_size={bs:>3}:  {100*zero/4000:5.1f}% of batches have NO sync negative")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------
# Prototype fix: a MoCo-style negative queue.
#
# The floor exists because a batch of 4 with binary labels is trivially satisfiable:
# two clusters separate four points, after which the gradient is zero. Enlarging the
# batch is the textbook fix, but a T4 at 224px/16 frames with gradient checkpointing
# cannot hold more than ~4 clips. A queue decouples the number of negatives from the
# batch: anchors come from the live batch (so the encoder still gets gradient), while
# positives and negatives come from a FIFO of recent detached embeddings.
# ---------------------------------------------------------------------------------
def supcon_with_queue(feat, labels, q_feat, q_labels, temperature: float = 0.1):
    """SupCon where the contrast set is the batch plus a queue of past embeddings."""
    import torch.nn.functional as F
    f = F.normalize(feat.float(), dim=-1)
    bank = F.normalize(torch.cat([feat.detach().float(), q_feat.float()]), dim=-1)
    bank_labels = torch.cat([labels, q_labels])
    sim = f @ bank.t() / temperature
    B = f.size(0)
    self_mask = torch.zeros_like(sim, dtype=torch.bool)
    self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=f.device)
    pos = (labels.unsqueeze(1) == bank_labels.unsqueeze(0)) & ~self_mask
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return feat.sum() * 0.0
    return -(log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / n_pos[valid]).mean()


def queue_demo(B=4, queue_size=1024, steps=600, dim=128, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    pool = 2048
    emb = torch.nn.Parameter(torch.randn(pool, dim) * 0.5)
    pool_lab = torch.stack([qacp_sync_labels(1, g) for _ in range(pool)]).squeeze(1)
    opt = torch.optim.Adam([emb], lr=lr)
    q_f = torch.randn(queue_size, dim) * 0.5
    q_l = pool_lab[torch.randint(0, pool, (queue_size,), generator=g)]
    loss = gnorm = 0.0
    for _ in range(steps):
        sel = torch.randint(0, pool, (B,), generator=g)
        l = supcon_with_queue(emb[sel], pool_lab[sel], q_f, q_l)
        opt.zero_grad()
        l.backward()
        gnorm = float(emb.grad.norm())
        opt.step()
        loss = float(l.detach())
        q_f = torch.cat([emb[sel].detach(), q_f])[:queue_size]      # FIFO
        q_l = torch.cat([pool_lab[sel], q_l])[:queue_size]
    return loss, gnorm


def queue_section():
    print("\n--- prototype fix: negative queue, batch stays at 4 ---")
    for qs in (0, 256, 1024, 4096):
        if qs == 0:
            loss, gnorm, _ = optimise(qacp_sync_labels, B=4)
            print(f"  queue={qs:>5} (current)  final sync loss {loss:>8.4f}  gnorm {gnorm:>8.4f}")
        else:
            loss, gnorm = queue_demo(B=4, queue_size=qs)
            print(f"  queue={qs:>5}            final sync loss {loss:>8.4f}  gnorm {gnorm:>8.4f}")
    print("  a loss well above ln(3)=1.0986 with non-zero gnorm = the term is still learning")



if __name__ == "__main__":
    main()
    queue_section()