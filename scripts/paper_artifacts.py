"""Build every manuscript artifact from a finished run and push it to HF.

Inputs (all produced by the Kaggle notebook / HF backup):
    eval JSONs        eval_<run_id>_<dataset>.json   (src/eval/evaluate.py reports, incl. preds)
    robustness JSON   robustness_<run_id>.json       (src/eval/robustness.py; optional)
    training logs     runs/<run_id>/logs/train_log.jsonl on HF (Stage-1 + QACP)
    configs           stage1_*_config.yaml, qacp_config.yaml

Outputs (local `<out>/` and HF `paper/` in the backup repo):
    metrics/summary.json          per dataset x seed metrics + mean/std + bootstrap 95% CI
    metrics/per_generator.json    in-domain per-generator/per-quadrant AUC/EER/acc per seed
    metrics/efficiency.json       params, trainable params, checkpoint size, T4 latency
    metrics/tables.tex            LaTeX table bodies (Table 1 in-domain, Table 2 cross-dataset,
                                  per-generator, efficiency) -> paste into report/chapters
    figures/results_roc*.pdf/png  ROC in-domain (3 seeds) + cross-dataset
    figures/results_reliability, results_confusion, results_localization, results_robustness
    figures/results_training_curves.pdf/png  (loss + val AUC per epoch, all seeds; QACP loss)
    figures/results_per_generator.pdf/png
    logs/*.jsonl                  copies of every train_log.jsonl
    configs/*.yaml

Usage (Kaggle Cell 13):
    python scripts/paper_artifacts.py --work /kaggle/working --out /kaggle/working/paper \
        --run-ids stage1_v2_seed42 stage1_v2_seed123 stage1_v2_seed456 --qacp-run-id qacp_stage0_v2 \
        --in-domain fakeavceleb-test --push
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ID = "MoshinAli/david-net-av-backup"
EVAL_RE = re.compile(r"^eval_(?P<run>.+?)_(?P<ds>[a-z0-9\-]+)\.json$")


# ------------------------------------------------------------------ helpers
def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _fmt(m, s=None, digits=3):
    if _nan(m):
        return "--"
    return f"{m:.{digits}f}" if s is None or _nan(s) else f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def _bootstrap_auc(y_true, y_score, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return None, None
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def collect_evals(work: Path, run_ids: list[str], hf_token: str | None) -> dict:
    """{(run_id, dataset): report}. Local files first, then HF runs/<run_id>/eval/."""
    reports = {}
    for p in sorted(work.glob("eval_*.json")):
        m = EVAL_RE.match(p.name)
        if m and m.group("run") in run_ids:
            reports[(m.group("run"), m.group("ds"))] = _load(p)
    if hf_token:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=hf_token)
        for run in run_ids:
            try:
                files = api.list_repo_tree(REPO_ID, path_in_repo=f"runs/{run}/eval", repo_type="model")
            except Exception:
                continue
            for f in files:
                if not f.path.endswith(".json"):
                    continue
                ds = Path(f.path).stem
                if (run, ds) in reports:
                    continue
                local = hf_hub_download(REPO_ID, f.path, repo_type="model", token=hf_token,
                                        local_dir=str(work / "hf_dl"))
                reports[(run, ds)] = _load(Path(local))
    return reports


def collect_logs(work: Path, run_ids: list[str], hf_token: str | None, out: Path) -> dict:
    """{run_id: [log entries]} from local train_log.jsonl or HF runs/<run>/logs/."""
    logs = {}
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if hf_token:
        from huggingface_hub import hf_hub_download
        for run in run_ids:
            try:
                local = hf_hub_download(REPO_ID, f"runs/{run}/logs/train_log.jsonl", repo_type="model",
                                        token=hf_token, local_dir=str(work / "hf_dl"))
                entries = [json.loads(l) for l in Path(local).read_text().splitlines() if l.strip()]
                logs[run] = entries
                shutil.copy2(local, out / "logs" / f"{run}_train_log.jsonl")
            except Exception as e:  # noqa: BLE001
                print(f"  no HF log for {run}: {e}")
    return logs


# ------------------------------------------------------------------ metrics
def summarize(reports: dict, run_ids: list[str], in_domain: str) -> dict:
    datasets = sorted({ds for _, ds in reports})
    summary = {"datasets": {}, "run_ids": run_ids, "in_domain": in_domain}
    for ds in datasets:
        per_seed = {}
        for run in run_ids:
            r = reports.get((run, ds))
            if r is None:
                continue
            per_seed[run] = {
                "n": r["n"],
                "video_auc": r["video"]["auc"], "video_eer": r["video"]["eer"],
                "video_f1": r["video"]["f1"], "video_acc": r["video"]["acc"],
                "audio_auc": r["audio"]["auc"], "audio_eer": r["audio"]["eer"],
                "audio_f1": r["audio"]["f1"], "audio_acc": r["audio"]["acc"],
                "quadrant_acc": r["quadrant"]["acc"], "quadrant_macro_f1": r["quadrant"]["macro_f1"],
                "video_ece": r["calibration"]["video_ece"], "audio_ece": r["calibration"]["audio_ece"],
            }
        if not per_seed:
            continue
        keys = next(iter(per_seed.values())).keys()
        agg = {}
        for k in keys:
            vals = [v[k] for v in per_seed.values() if not _nan(v[k])]
            agg[k] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                      "std": float(np.std(vals)) if vals else float("nan"), "n_seeds": len(vals)}
        # bootstrap CI on the first seed's predictions
        first = reports.get((run_ids[0], ds)) or next(iter(reports[(r, ds)] for r in per_seed))
        ci = {}
        for mod in ("video", "audio"):
            p = first["preds"][mod]
            ci[mod] = _bootstrap_auc(p["y_true"], p["y_score"])
        summary["datasets"][ds] = {"per_seed": per_seed, "aggregate": agg, "auc_ci95_seed0": ci,
                                   "confusion_seed0": first["quadrant"].get("confusion")}
    return summary


def per_generator_table(reports: dict, run_ids: list[str], in_domain: str) -> dict:
    out = {"video": defaultdict(list), "audio": defaultdict(list), "quadrant": defaultdict(list)}
    for run in run_ids:
        r = reports.get((run, in_domain))
        if not r or "per_generator" not in r:
            continue
        for mod in ("video", "audio"):
            for g, m in r["per_generator"][mod].items():
                out[mod][g].append(m)
            for q, m in r.get("per_quadrant", {}).get(mod, {}).items():
                out["quadrant"][f"{mod}:{q}"].append(m)
    table = {}
    for mod, groups in out.items():
        table[mod] = {}
        for g, ms in groups.items():
            table[mod][g] = {"n": ms[0]["n"], "pos_rate": ms[0]["pos_rate"]}
            for k in ("auc", "eer", "acc"):
                vals = [m[k] for m in ms if not _nan(m[k])]
                table[mod][g][k] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                                    "std": float(np.std(vals)) if vals else float("nan")}
    return table


def efficiency(config_path: Path | None, checkpoint: Path | None) -> dict:
    info = {}
    if checkpoint and checkpoint.exists():
        info["checkpoint_gb"] = round(checkpoint.stat().st_size / 1e9, 3)
    if config_path and config_path.exists():
        try:
            import torch
            from src.utils.config import load_config
            from src.training.train import build_model
            cfg = load_config(str(config_path))
            model = build_model(cfg)
            info["params_total"] = sum(p.numel() for p in model.parameters())
            info["params_trainable_stage1"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev).eval()
            v = torch.rand(1, cfg.n_frames, 3, 224, 224, device=dev)
            a = torch.rand(1, cfg.audio_len, device=dev)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                for _ in range(3):
                    model(v, a)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                n = 10
                for _ in range(n):
                    model(v, a)
                if dev == "cuda":
                    torch.cuda.synchronize()
            info["latency_ms_per_clip"] = round((time.time() - t0) / n * 1000, 1)
            info["latency_device"] = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
            if dev == "cuda":
                info["peak_vram_gb_inference"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception as e:  # noqa: BLE001
            info["error"] = str(e)[:200]
    return info


# ------------------------------------------------------------------ extra experiments
CROSS_FOR_ABLATION = {"celeb-df-v2": "video_auc", "in-the-wild": "audio_auc"}   # dataset -> the stream it can score
ROW_END = " \\\\"


def _mean_auc(r: dict) -> float:
    vals = [r["video"]["auc"], r["audio"]["auc"]]
    vals = [v for v in vals if not _nan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def ablation_rows(reports: dict, main_run_ids: list[str], abl_run_ids: list[str], in_domain: str) -> list[dict]:
    """rows for fig_ablation / Table 5: in-domain AUC (mean of V/A) and cross-dataset AUC
    (video AUC on Celeb-DF + audio AUC on In-the-Wild, averaged over what exists)."""
    def _row(name, runs):
        ind, cross = [], []
        for run in runs:
            r = reports.get((run, in_domain))
            if r:
                ind.append(_mean_auc(r))
            for ds, key in CROSS_FOR_ABLATION.items():
                rr = reports.get((run, ds))
                if rr:
                    v = rr["video" if key == "video_auc" else "audio"]["auc"]
                    if not _nan(v):
                        cross.append(v)
        return {"name": name, "n_runs": len(runs),
                "in_domain_auc": float(np.mean(ind)) if ind else None,
                "cross_dataset_auc": float(np.mean(cross)) if cross else None}
    rows = [_row("full (3 seeds)", main_run_ids)]
    for run in abl_run_ids:
        name = run.replace("abl_", "").split("_v2")[0].split("_seed")[0]
        rows.append(_row(name, [run]))
    return rows


def logo_rows(reports: dict, logo_run_ids: list[str]) -> list[dict]:
    rows = []
    for run in logo_run_ids:
        fam = run.replace("logo_", "").split("_v2")[0].split("_seed")[0]
        r = reports.get((run, f"logo-{fam}"))
        if not r:
            continue
        rows.append({"family": fam, "n": r["n"], "video_auc": r["video"]["auc"], "video_eer": r["video"]["eer"],
                     "audio_auc": r["audio"]["auc"], "audio_eer": r["audio"]["eer"],
                     "quadrant_acc": r["quadrant"]["acc"]})
    return rows


def fairness_table(reports: dict, run_ids: list[str], in_domain: str) -> dict:
    """Table 6: per-race / per-gender AUC + EER (mean over seeds) and the max subgroup gap."""
    out = {}
    for attr in ("race", "gender"):
        acc = defaultdict(lambda: defaultdict(list))
        for run in run_ids:
            r = reports.get((run, in_domain))
            fair = (r or {}).get("fairness") or {}
            for mod in ("video", "audio"):
                for g, m in fair.get(attr, {}).get(mod, {}).items():
                    if g == "":
                        continue
                    acc[mod][g].append(m)
        table = {}
        for mod, groups in acc.items():
            table[mod] = {}
            for g, ms in groups.items():
                table[mod][g] = {"n": ms[0]["n"]}
                for k in ("auc", "eer"):
                    vals = [m[k] for m in ms if not _nan(m[k])]
                    table[mod][g][k] = float(np.mean(vals)) if vals else float("nan")
            aucs = [v["auc"] for v in table[mod].values() if not _nan(v["auc"])]
            table[mod]["_max_auc_gap"] = float(max(aucs) - min(aucs)) if len(aucs) > 1 else float("nan")
        out[attr] = table
    return out


def collect_baselines(work: Path) -> list[dict]:
    return [_load(p) for p in sorted(work.glob("baseline_*.json"))]


# ------------------------------------------------------------------ tables
def latex_tables(summary: dict, per_gen: dict, eff: dict, in_domain: str,
                 baselines: list[dict] | None = None, ablations: list[dict] | None = None,
                 logo: list[dict] | None = None, fairness: dict | None = None) -> str:
    L = []
    ds_in = summary["datasets"].get(in_domain)
    L.append("% ===== Table 1: in-domain (FakeAVCeleb test, subject-disjoint), mean +- std over seeds")
    L.append("% Method & V-AUC & V-EER & A-AUC & A-EER & Quad-Acc & Quad-F1 & V-ECE & A-ECE \\\\")
    if ds_in:
        a = ds_in["aggregate"]
        L.append("DAVID-Net (ours) & " + " & ".join([
            _fmt(a["video_auc"]["mean"], a["video_auc"]["std"]), _fmt(a["video_eer"]["mean"], a["video_eer"]["std"]),
            _fmt(a["audio_auc"]["mean"], a["audio_auc"]["std"]), _fmt(a["audio_eer"]["mean"], a["audio_eer"]["std"]),
            _fmt(a["quadrant_acc"]["mean"], a["quadrant_acc"]["std"]), _fmt(a["quadrant_macro_f1"]["mean"], a["quadrant_macro_f1"]["std"]),
            _fmt(a["video_ece"]["mean"], a["video_ece"]["std"]), _fmt(a["audio_ece"]["mean"], a["audio_ece"]["std"]),
        ]) + " \\\\")
        ci = ds_in["auc_ci95_seed0"]
        L.append(f"% 95% bootstrap CI (seed 0): video AUC {ci['video']}, audio AUC {ci['audio']}")
    for b in baselines or []:
        mod = b.get("modality"); m = b["metrics"][mod]
        cells = ["--"] * 8
        if mod == "video":
            cells[0], cells[1] = _fmt(m["auc"]), _fmt(m["eer"])
        else:
            cells[2], cells[3] = _fmt(m["auc"]), _fmt(m["eer"])
        L.append(f"{b['method']} (baseline, 1 seed) & " + " & ".join(cells) + ROW_END)
    L.append("")
    L.append("% ===== Table 2: cross-dataset (trained on FakeAVCeleb train split only)")
    L.append("% Dataset & Modality & n & V-AUC & A-AUC & Quad-Acc \\\\")
    for ds, d in summary["datasets"].items():
        if ds == in_domain:
            continue
        a = d["aggregate"]
        n = next(iter(d["per_seed"].values()))["n"]
        L.append(f"{ds} & & {n} & {_fmt(a['video_auc']['mean'], a['video_auc']['std'])} & "
                 f"{_fmt(a['audio_auc']['mean'], a['audio_auc']['std'])} & "
                 f"{_fmt(a['quadrant_acc']['mean'], a['quadrant_acc']['std'])} \\\\")
    L.append("")
    L.append("% ===== Per-generator breakdown on the in-domain test split (video head / audio head)")
    L.append("% Generator & n & V-AUC & V-Acc & A-AUC & A-Acc \\\\")
    gens = sorted(set(per_gen.get("video", {})) | set(per_gen.get("audio", {})))
    for g in gens:
        v = per_gen.get("video", {}).get(g, {}); a = per_gen.get("audio", {}).get(g, {})
        L.append(f"{g} & {v.get('n', a.get('n', ''))} & {_fmt(v.get('auc', {}).get('mean'), v.get('auc', {}).get('std'))} & "
                 f"{_fmt(v.get('acc', {}).get('mean'), v.get('acc', {}).get('std'))} & "
                 f"{_fmt(a.get('auc', {}).get('mean'), a.get('auc', {}).get('std'))} & "
                 f"{_fmt(a.get('acc', {}).get('mean'), a.get('acc', {}).get('std'))} \\\\")
    L.append("")
    if ablations:
        L.append("% ===== Table 5: ablations (seed 42, reduced epochs unless noted; full = 3-seed mean)")
        L.append("% Variant & In-domain AUC (mean V/A) & Cross-dataset AUC" + ROW_END)
        for r in ablations:
            L.append(f"{r['name']} & {_fmt(r['in_domain_auc'])} & {_fmt(r['cross_dataset_auc'])}" + ROW_END)
        L.append("")
    if logo:
        L.append("% ===== Table 3: leave-one-generator-family-out (subject-disjoint test identities)")
        L.append("% Held-out family & n & V-AUC & V-EER & A-AUC & A-EER & Quad-Acc" + ROW_END)
        for r in logo:
            L.append(f"{r['family']} & {r['n']} & {_fmt(r['video_auc'])} & {_fmt(r['video_eer'])} & "
                     f"{_fmt(r['audio_auc'])} & {_fmt(r['audio_eer'])} & {_fmt(r['quadrant_acc'])}" + ROW_END)
        L.append("")
    if fairness:
        L.append("% ===== Table 6: fairness subgroup gaps on the in-domain test split (mean over seeds)")
        L.append("% Attribute & Group & n & V-AUC & A-AUC" + ROW_END)
        for attr, table in fairness.items():
            groups = sorted(set(table.get("video", {})) | set(table.get("audio", {})))
            for g in groups:
                if g.startswith("_"):
                    continue
                v = table.get("video", {}).get(g, {}); a = table.get("audio", {}).get(g, {})
                L.append(f"{attr} & {g} & {v.get('n', a.get('n', ''))} & {_fmt(v.get('auc'))} & {_fmt(a.get('auc'))}" + ROW_END)
            L.append(f"% {attr}: max AUC gap video={_fmt(table.get('video', {}).get('_max_auc_gap'))} "
                     f"audio={_fmt(table.get('audio', {}).get('_max_auc_gap'))}")
        L.append("")
    L.append("% ===== Table 7: efficiency")
    L.append(f"% params total = {eff.get('params_total', '?')}, trainable (Stage 1) = {eff.get('params_trainable_stage1', '?')}, "
             f"checkpoint = {eff.get('checkpoint_gb', '?')} GB, latency = {eff.get('latency_ms_per_clip', '?')} ms/clip on {eff.get('latency_device', '?')}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ figures
def fig_training_curves(logs: dict, qacp_log: list | None, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_panels = 3 if qacp_log else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.3 * n_panels, 2.9))
    for run, entries in logs.items():
        ep = [e["epoch"] for e in entries if "avg_loss" in e]
        axes[0].plot(ep, [e["avg_loss"] for e in entries if "avg_loss" in e], label=run.replace("stage1_", ""))
        if any("video_auc" in e for e in entries):
            axes[1].plot(ep, [e.get("video_auc", np.nan) for e in entries if "avg_loss" in e], linestyle="-", label=f"{run.split('seed')[-1]} video")
            axes[1].plot(ep, [e.get("audio_auc", np.nan) for e in entries if "avg_loss" in e], linestyle="--", label=f"{run.split('seed')[-1]} audio")
    axes[0].set_title("Stage-1 training loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=6)
    axes[1].set_title("Validation AUC"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0.4, 1.02); axes[1].legend(fontsize=6)
    if qacp_log:
        ep = [e["epoch"] for e in qacp_log if "avg_loss" in e]
        axes[2].plot(ep, [e["avg_loss"] for e in qacp_log if "avg_loss" in e], label="total")
        for k in ("qacp_v", "qacp_a", "qacp_c"):
            if any(k in e for e in qacp_log):
                axes[2].plot(ep, [e.get(k, np.nan) for e in qacp_log if "avg_loss" in e], linestyle="--", label=k)
        axes[2].set_title("QACP (Stage 0) loss"); axes[2].set_xlabel("epoch"); axes[2].legend(fontsize=6)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "results_training_curves.pdf"); fig.savefig(out_dir / "results_training_curves.png", dpi=200)
    plt.close(fig)


def fig_per_generator(per_gen: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gens = [g for g in sorted(set(per_gen.get("video", {})) | set(per_gen.get("audio", {}))) if g != "real"]
    if not gens:
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(gens)); w = 0.38
    for i, mod in enumerate(("video", "audio")):
        vals = [per_gen.get(mod, {}).get(g, {}).get("acc", {}).get("mean", np.nan) for g in gens]
        errs = [per_gen.get(mod, {}).get(g, {}).get("acc", {}).get("std", 0.0) for g in gens]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=2, label=f"{mod} head accuracy")
    ax.set_xticks(x); ax.set_xticklabels(gens, rotation=20, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy on generator's clips"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "results_per_generator.pdf"); fig.savefig(out_dir / "results_per_generator.png", dpi=200)
    plt.close(fig)


def build_figures(reports: dict, run_ids: list[str], in_domain: str, robustness: Path | None, out: Path,
                  baselines: list[dict] | None = None, ablations: list[dict] | None = None):
    """Run src.eval.figures on a staged results dir (in-domain seeds -> ROC/reliability/
    confusion/localization; cross-dataset -> a second ROC)."""
    from src.eval.figures import generate_all, fig_roc
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stage = out / "_stage_in_domain"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for i, run in enumerate(run_ids):
        r = reports.get((run, in_domain))
        if r:
            r = dict(r); r["method"] = f"david-net seed {run.split('seed')[-1]}"
            (stage / f"david-net_seed{i}.json").write_text(json.dumps(r), encoding="utf-8")
    if robustness and robustness.exists():
        shutil.copy2(robustness, stage / "robustness_david-net.json")
    for b in baselines or []:                       # baselines share the ROC panel
        (stage / f"baseline_{b['method']}.json").write_text(json.dumps(b), encoding="utf-8")
    if ablations:
        (stage / "ablation.json").write_text(json.dumps(ablations), encoding="utf-8")
    if any(stage.iterdir()):
        generate_all(str(stage), str(fig_dir), demo=False)
    # cross-dataset ROC (first seed), one curve per dataset
    cross = []
    for (run, ds), r in reports.items():
        if run == run_ids[0] and ds != in_domain:
            rr = dict(r); rr["method"] = ds
            cross.append(rr)
    if cross:
        fig_roc(cross, fig_dir)
        for ext in ("pdf", "png"):
            src = fig_dir / f"results_roc.{ext}"
            if src.exists():
                shutil.move(str(src), str(fig_dir / f"results_roc_cross_dataset.{ext}"))
        # regenerate the in-domain ROC (fig_roc overwrote it)
        if any(stage.glob("david-net_seed*.json")):
            fig_roc([_load(p) for p in sorted(stage.glob("david-net_seed*.json"))]
                    + [_load(p) for p in sorted(stage.glob("baseline_*.json"))], fig_dir)
    shutil.rmtree(stage, ignore_errors=True)


# ------------------------------------------------------------------ upload
def push(out: Path, hf_token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_folder(folder_path=str(out), path_in_repo="paper", repo_id=REPO_ID, repo_type="model",
                      commit_message="paper artifacts: metrics, tables, figures, logs, configs")
    print(f"pushed {sum(1 for _ in out.rglob('*') if _.is_file())} files to {REPO_ID}/paper/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--out", default="/kaggle/working/paper")
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--qacp-run-id", default=None)
    ap.add_argument("--in-domain", default="fakeavceleb-test")
    ap.add_argument("--robustness", default=None, help="robustness JSON (optional)")
    ap.add_argument("--config", default=None, help="a Stage-1 config yaml (for efficiency numbers)")
    ap.add_argument("--checkpoint", default=None, help="best.pt (for checkpoint size)")
    ap.add_argument("--ablation-run-ids", nargs="*", default=[], help="abl_<name>_v2_seed42 ...")
    ap.add_argument("--logo-run-ids", nargs="*", default=[], help="logo_<family>_v2_seed42 ...")
    ap.add_argument("--explain-dir", default=None, help="dir with results_explain_*.{pdf,png}")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True); (out / "configs").mkdir(exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("hf")

    reports = collect_evals(work, args.run_ids + args.ablation_run_ids + args.logo_run_ids, token)
    print(f"{len(reports)} eval reports:", sorted(reports))
    if not reports:
        raise SystemExit("no eval reports found — run the evaluation cell first")
    logs = collect_logs(work, args.run_ids, token, out)
    qacp_log = collect_logs(work, [args.qacp_run_id], token, out).get(args.qacp_run_id) if args.qacp_run_id else None

    summary = summarize(reports, args.run_ids, args.in_domain)
    per_gen = per_generator_table(reports, args.run_ids, args.in_domain)
    eff = efficiency(Path(args.config) if args.config else None, Path(args.checkpoint) if args.checkpoint else None)
    baselines = collect_baselines(work)
    ablations = ablation_rows(reports, args.run_ids, args.ablation_run_ids, args.in_domain) if args.ablation_run_ids else None
    logo = logo_rows(reports, args.logo_run_ids) if args.logo_run_ids else None
    fairness = fairness_table(reports, args.run_ids, args.in_domain)
    (out / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "metrics" / "per_generator.json").write_text(json.dumps(per_gen, indent=2), encoding="utf-8")
    (out / "metrics" / "efficiency.json").write_text(json.dumps(eff, indent=2), encoding="utf-8")
    (out / "metrics" / "fairness.json").write_text(json.dumps(fairness, indent=2), encoding="utf-8")
    if baselines:
        (out / "metrics" / "baselines.json").write_text(json.dumps(
            [{k: v for k, v in b.items() if k != "preds"} for b in baselines], indent=2), encoding="utf-8")
    if ablations:
        (out / "metrics" / "ablation.json").write_text(json.dumps(ablations, indent=2), encoding="utf-8")
    if logo:
        (out / "metrics" / "logo.json").write_text(json.dumps(logo, indent=2), encoding="utf-8")
    (out / "metrics" / "tables.tex").write_text(
        latex_tables(summary, per_gen, eff, args.in_domain, baselines, ablations, logo, fairness), encoding="utf-8")
    if args.explain_dir and Path(args.explain_dir).exists():
        (out / "figures").mkdir(parents=True, exist_ok=True)
        for f in Path(args.explain_dir).glob("results_explain_*"):
            shutil.copy2(f, out / "figures" / f.name)
        if (Path(args.explain_dir) / "explain_index.json").exists():
            shutil.copy2(Path(args.explain_dir) / "explain_index.json", out / "metrics" / "explain_index.json")
    # raw per-(seed, dataset) reports without the prediction dumps + full dumps separately
    (out / "metrics" / "eval").mkdir(exist_ok=True)
    for (run, ds), r in reports.items():
        slim = {k: v for k, v in r.items() if k != "preds"}
        (out / "metrics" / "eval" / f"{run}_{ds}.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
        (out / "metrics" / "eval" / f"{run}_{ds}_preds.json").write_text(json.dumps(r["preds"]), encoding="utf-8")
    for cfgp in list(work.glob("stage1_*config.yaml")) + list(work.glob("qacp_config.yaml")):
        shutil.copy2(cfgp, out / "configs" / cfgp.name)

    build_figures(reports, args.run_ids, args.in_domain, Path(args.robustness) if args.robustness else None, out,
                  baselines=baselines, ablations=ablations)
    if logs:
        fig_training_curves(logs, qacp_log, out / "figures")
    fig_per_generator(per_gen, out / "figures")
    if args.robustness and Path(args.robustness).exists():
        shutil.copy2(args.robustness, out / "metrics" / "robustness.json")

    print("\n" + (out / "metrics" / "tables.tex").read_text(encoding="utf-8"))
    print("artifacts:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and "_preds" not in p.name))
    if args.push:
        if not token:
            raise SystemExit("HF_TOKEN missing")
        push(out, token)


if __name__ == "__main__":
    main()
