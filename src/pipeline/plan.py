"""The experiment matrix: Baseline -> QACP -> Proposed -> Ablations -> LOGO -> Legacy -> Final.

Every entry is a separate experiment (own EXP id, config, checkpoint, metrics) and
every one is run for every seed in CONFIG['seeds']. Relationships are explicit:
`init_from` names the experiment whose best weights warm-start this one (same seed),
and `components` records which proposed components are ON, which is what the ablation
table is built from -- nothing about an improvement is claimed by construction; the
tables show what the measurements say, with DeLong tests against the proposed model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.pipeline.config import build_config, config_hash, deep_merge, set_dotted

COMPONENTS = ("qacp", "sync", "disentangle", "localization", "multitask", "modality_dropout",
              "mismatch_class", "copy_synthesis", "self_blending")
FULL = {c: True for c in COMPONENTS}


@dataclass
class ExpSpec:
    name: str
    stage: str                       # qacp | stage1 | baseline | phase_b
    group: str                       # baseline | qacp | proposed | ablation | logo | legacy | final
    split: str = "strict"            # strict | legacy | logo_<family>
    overrides: dict = field(default_factory=dict)      # dotted keys
    init_from: Optional[str] = None  # experiment name (same seed)
    components: dict = field(default_factory=lambda: dict(FULL))
    description: str = ""
    baseline_model: Optional[str] = None
    evaluate_cross: bool = True


def _qacp(name: str, desc: str, overrides=None, comps=None, split="strict") -> ExpSpec:
    return ExpSpec(name=name, stage="qacp", group="qacp", split=split,
                   overrides=overrides or {}, components=dict(FULL, **(comps or {})),
                   description=desc, evaluate_cross=False)


def _s1(name: str, group: str, desc: str, overrides=None, init="qacp", comps=None,
        split="strict", cross=True) -> ExpSpec:
    return ExpSpec(name=name, stage="stage1", group=group, split=split,
                   overrides=overrides or {}, init_from=init,
                   components=dict(FULL, **(comps or {})), description=desc,
                   evaluate_cross=cross)


def full_plan() -> list[ExpSpec]:
    nq = ["RVRA", "RVFA", "FVRA", "FVFA"]
    p = [
        # ------------------------------------------------------------ baselines
        ExpSpec("video-probe", "baseline", "baseline", baseline_model="video_probe",
                components={c: False for c in COMPONENTS},
                description="Linear probe on mean-pooled frozen VideoMAE tokens (video only)."),
        ExpSpec("audio-probe", "baseline", "baseline", baseline_model="audio_probe",
                components={c: False for c in COMPONENTS},
                description="Linear probe on mean-pooled frozen WavLM tokens (audio only)."),
        ExpSpec("late-fusion", "baseline", "baseline", baseline_model="late_fusion",
                components={c: False for c in COMPONENTS},
                description="Concatenated pooled embeddings + MLP heads; no cross-attention, "
                            "sync, disentanglement or pretraining."),
        # ------------------------------------------------------------ QACP pretraining
        _qacp("qacp", "QACP Stage 0, all five pseudo classes."),
        _qacp("qacp-no_sync", "QACP without the sync module (feeds the no-sync ablation).",
              {"model.use_sync": False}, {"sync": False}),
        _qacp("qacp-no_mismatch", "QACP without the MISMATCH pseudo class.",
              {"train.qacp.pseudo_classes": nq}, {"mismatch_class": False}),
        _qacp("qacp-no_copysynth", "QACP without copy-synthesised (pseudo-fake) audio.",
              {"train.qacp.pseudo_classes": ["RVRA", "FVRA", "MISMATCH"]}, {"copy_synthesis": False}),
        _qacp("qacp-no_selfblend", "QACP without self-blended (pseudo-fake) video.",
              {"train.qacp.pseudo_classes": ["RVRA", "RVFA", "MISMATCH"]}, {"self_blending": False}),
        # ------------------------------------------------------------ proposed
        _s1("davidnet", "proposed", "DAVID-Net, QACP-initialised, all components (Phase A)."),
        # ------------------------------------------------------------ ablations
        _s1("davidnet-no_qacp", "ablation", "No QACP pretraining (random init).",
            init=None, comps={"qacp": False}),
        _s1("davidnet-no_sync", "ablation", "Without the sync module.",
            {"model.use_sync": False, "train.stage1.loss_weights.sync": 0.0},
            init="qacp-no_sync", comps={"sync": False}),
        _s1("davidnet-no_disentangle", "ablation", "Without the disentanglement loss.",
            {"model.use_disentangle": False, "train.stage1.loss_weights.disentangle": 0.0},
            comps={"disentangle": False}),
        _s1("davidnet-no_loc", "ablation", "Without the localization loss.",
            {"train.stage1.loss_weights.loc": 0.0}, comps={"localization": False}),
        _s1("davidnet-single_task", "ablation", "Per-modality heads only (no quad/loc/sync/dis. losses).",
            {"train.stage1.loss_weights.quad": 0.0, "train.stage1.loss_weights.loc": 0.0,
             "train.stage1.loss_weights.sync": 0.0, "train.stage1.loss_weights.disentangle": 0.0},
            comps={"multitask": False, "localization": False}),
        _s1("davidnet-no_moddrop", "ablation", "Without modality dropout.",
            {"train.stage1.modality_dropout": 0.0}, comps={"modality_dropout": False}),
        _s1("davidnet-compose_quad", "ablation", "Quadrant composed from H_v x H_a instead of a head.",
            {"model.compose_quadrant": True}),
        _s1("davidnet-qacp_no_mismatch", "ablation", "QACP without the MISMATCH class.",
            init="qacp-no_mismatch", comps={"mismatch_class": False}),
        _s1("davidnet-qacp_no_copysynth", "ablation", "QACP without copy-synthesis.",
            init="qacp-no_copysynth", comps={"copy_synthesis": False}),
        _s1("davidnet-qacp_no_selfblend", "ablation", "QACP without self-blending.",
            init="qacp-no_selfblend", comps={"self_blending": False}),
    ]
    # ---------------------------------------------------------------- LOGO (video families)
    for fam in ("wav2lip", "fsgan", "faceswap"):
        p.append(_s1(f"davidnet-logo_{fam}", "logo",
                     f"Leave-one-generator-family-out: {fam} unseen in training/validation.",
                     split=f"logo_{fam}", cross=False))
    # ---------------------------------------------------------------- legacy protocol
    p.append(_qacp("qacp-legacy", "QACP on the legacy (source-identity) training reals.",
                   split="legacy"))
    p.append(_s1("davidnet-legacy", "legacy",
                 "DAVID-Net on the literature's source-identity split (target identities leak; "
                 "supplementary comparability table only).", init="qacp-legacy", split="legacy",
                 cross=False))
    # ---------------------------------------------------------------- final (Phase B)
    p.append(ExpSpec("davidnet-e2e", "phase_b", "final", init_from="davidnet",
                     description="Final model: Phase-A DAVID-Net fine-tuned end to end "
                                 "(top encoder blocks unfrozen)."))
    return p


def select(plan: list[ExpSpec], groups: Optional[list] = None,
           only: Optional[list] = None) -> list[ExpSpec]:
    keep = [s for s in plan if (groups is None or s.group in groups)]
    if only:
        keep = [s for s in keep if s.name in only]
    names = {s.name for s in keep}
    # pull in dependencies (a stage-1 run cannot start without its QACP init)
    changed = True
    while changed:
        changed = False
        for s in plan:
            if s.name not in names and any(k.init_from == s.name for k in keep):
                keep.append(s)
                names.add(s.name)
                changed = True
    order = {s.name: i for i, s in enumerate(plan)}
    return sorted(keep, key=lambda s: order[s.name])


def resolve(base_cfg: dict, spec: ExpSpec, seed: int) -> dict:
    """Experiment config = base CONFIG + spec overrides + seed + the spec itself.

    The spec (stage, split, relationship, components) is part of the semantic config,
    so two experiments that differ only in, e.g., their init are different hashes.
    """
    cfg = deep_merge(base_cfg, {})
    for k, v in spec.overrides.items():
        set_dotted(cfg, k, v)
    cfg["seed"] = seed
    cfg["experiment"] = {"name": spec.name, "stage": spec.stage, "group": spec.group,
                         "split": spec.split, "init_from": spec.init_from,
                         "baseline_model": spec.baseline_model,
                         "components": spec.components, "description": spec.description}
    return cfg


def smoke_overrides(cfg: dict) -> dict:
    """Tiny schedule for mode=smoke (same code path, seconds per experiment)."""
    e = cfg["smoke"]["epochs"]
    over = {"train": {k: {"epochs": e, "early_stopping_patience": 99} for k in cfg["train"]},
            "checkpoint": {"every_steps": cfg["smoke"]["every_steps"], "every_minutes": 999.0},
            "evaluation": {"bootstrap": 50}}
    over["train"]["qacp"]["items_per_epoch"] = 128
    return over


def plan_table(plan: list[ExpSpec], seeds: list[int], base_cfg: dict) -> list[dict]:
    rows = []
    for s in plan:
        for seed in seeds:
            c = resolve(base_cfg, s, seed)
            rows.append({"name": s.name, "seed": seed, "stage": s.stage, "group": s.group,
                         "split": s.split, "init_from": s.init_from or "",
                         "config_hash": config_hash(c), "description": s.description})
    return rows
