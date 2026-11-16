"""The resumable session driver behind the Kaggle notebook.

Each Kaggle session does as much of the remaining work as fits in its time budget and
leaves everything it finished (and a resumable checkpoint of what it did not) on the
Hugging Face repo. Running the notebook again continues where the last session
stopped -- on the same account or another one (claims prevent double work).

    s = Session(CONFIG, repo_dir)
    s.connect()            # token, repos, registry
    s.prepare_data()       # frozen, hash-verified manifests + splits
    s.prepare_features()   # resumable feature extraction (+ Phase-B clip cache)
    s.run_experiments()    # plan order; each experiment trains/resumes, evaluates, uploads
    s.build_reports()      # aggregate tables + figures from saved results
    s.final_report()
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.pipeline import config as C
from src.pipeline.env import (Stopwatch, autoconfig_hardware, environment_report, log_event,
                              pip_freeze, quiet_third_party, seed_everything, setup_logging,
                              utcnow)
from src.pipeline.hub import HubStore, resolve_hf_token, sha256_file, validate_token

VIDEO_CORPORA = ("fakeavceleb", "celeb-df-v2", "dfdc-10", "deepfaketimit")


class _LazyTableDict(dict):
    """Dictionary that lazily loads table partitions only when accessed to conserve RAM."""

    def __init__(self, loaders: dict):
        super().__init__()
        self._loaders = loaders

    def __getitem__(self, key):
        if not super().__contains__(key) and key in self._loaders:
            self[key] = self._loaders[key]()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if not super().__contains__(key) and key in self._loaders:
            self[key] = self._loaders[key]()
        return super().get(key, default)

    def __contains__(self, key):
        return super().__contains__(key) or key in self._loaders


class Session:
    # ================================================================ setup
    def __init__(self, user_config: dict, repo_dir: str | Path = ".", namespace: Optional[str] = None):
        import os
        quiet_third_party()
        cfg = C.build_config(user_config)
        if cfg["session"].get("worker_name"):
            os.environ.setdefault("DAVIDNET_WORKER", cfg["session"]["worker_name"])
        if cfg["mode"] == "smoke":
            from src.pipeline.plan import smoke_overrides
            cfg = C.deep_merge(cfg, smoke_overrides(cfg))
        if cfg["mode"] == "recovery_test":
            from src.pipeline.synthetic import recovery_config
            cfg = recovery_config(cfg)
        self.cfg = cfg
        self.mode = cfg["mode"]
        self.ns = namespace if namespace is not None else ("" if self.mode == "full" else f"{self.mode}/")
        if self.mode == "recovery_test":
            # each recovery test is isolated under its own tag (a reference run and a
            # crash/resume run can then be compared side by side)
            self.ns += (cfg.get("recovery") or {}).get("tag", "default") + "/"
        self.repo_dir = Path(repo_dir)
        self.work = Path(cfg["project"]["work_dir"]) / self.mode
        self.scratch = Path(cfg["project"]["scratch_dir"]) / self.mode
        self.work.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.log_path = setup_logging(self.work / "logs")
        self.env = environment_report(self.repo_dir, worker_name=cfg.get("session", {}).get("worker_name"))
        self.hw = autoconfig_hardware(cfg)
        self.stopwatch = Stopwatch(cfg["session"]["time_budget_hours"],
                                   cfg["session"]["safety_minutes"])
        seed_everything(cfg["seeds"][0], cfg["hardware"]["deterministic"])
        self.store = self.data_store = self.registry = None
        self.records: dict = {}
        self.splits: dict = {}
        self.fsid: Optional[str] = None
        self.revisions: dict = {}
        self._tables: dict = {}
        self._freeze: Optional[str] = None
        if os.path.exists("/kaggle"):
            from src.pipeline.env import start_memory_ticker
            start_memory_ticker()
        run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local")
        self.run_type = run_type
        if run_type == "Interactive" and self.mode == "full":
            log_event("interactive_session_warning",
                      "INTERACTIVE Kaggle session: it dies when the browser tab closes or the "
                      "connection drops (status CANCEL_ACKNOWLEDGED, no traceback). For training "
                      "use Save Version -> Save & Run All (Commit).", logging.WARNING)
        log_event("session_started", f"run_type={run_type} mode={self.mode} device={self.hw['device']} "
                  f"({self.hw['gpu_name']}, {self.hw['precision']})",
                  git=self.env.get("git", {}).get("commit"))

    def connect(self):
        token = resolve_hf_token()
        p = self.cfg["project"]
        who = validate_token(token, p["hf_repo"])
        self.store = HubStore(p["hf_repo"], token, p["hf_repo_type"], p["hf_private"],
                              retries=self.cfg["checkpoint"]["upload_retries"])
        self.store.ensure_repo()
        data_repo = p.get("hf_data_repo") or p["hf_repo"] + "-data"
        self.data_store = HubStore(data_repo, token, "dataset", True,
                                   retries=self.cfg["checkpoint"]["upload_retries"])
        self.data_store.ensure_repo()
        from src.pipeline.registry import Registry
        self.registry = Registry(self.store, self.ns)
        self.hub_revision_at_start = self.store.head()
        log_event("hub_connected", f"{who['user']} -> {self.store.web_url()} "
                  f"(+ data {self.data_store.web_url()})")
        return who

    # ================================================================ data
    def prepare_data(self):
        """Frozen manifests + splits. Built once, then only ever re-downloaded and verified."""
        from src.pipeline import manifests as M
        from src.pipeline import splits as S
        idx_path = f"{self.ns}data/SPLITS_SHA256.json"
        local = self.work / "data"
        idx = self.store.read_json(idx_path)
        data_ns = self.ns
        if idx is None and self.mode == "recovery_test":
            idx = self._build_synthetic_data(local, M)
        elif idx is None and self.ns:
            # Fall back to root frozen split if sub-namespace does not have its own
            idx = self.store.read_json("data/SPLITS_SHA256.json")
            if idx is not None:
                data_ns = ""
                log_event("data_fallback", f"reusing root frozen split data/SPLITS_SHA256.json for namespace '{self.ns}'")
        if idx is None:
            log_event("data_build", "no frozen split on the Hub -> building from Kaggle mounts")
            idx = self._build_data(local, M, S)
        for rel, meta in idx["files"].items():
            dst = local / rel
            if not dst.exists() or sha256_file(dst) != meta["sha256"]:
                self.store.download(f"{data_ns}data/{rel}", self.work)
                src = self.work / data_ns / "data" / rel if data_ns else dst
                if src != dst:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), dst)
            if sha256_file(dst) != meta["sha256"]:
                raise RuntimeError(f"data/{rel} fails sha256 verification -- refusing to train "
                                   "on a split that differs from the frozen one")
        self._freeze = idx["freeze_sha"]
        for p in (local / "manifests").glob("*.jsonl"):
            self.records[p.stem] = M.read_jsonl(p)
        for proto_dir in (local / "splits").iterdir():
            self.splits[proto_dir.name] = {s: M.read_jsonl(proto_dir / f"{s}.jsonl")
                                           for s in ("train", "val", "test")
                                           if (proto_dir / f"{s}.jsonl").exists()}
        S.assert_no_leakage(self.splits["strict"])
        log_event("dataset_loaded", f"frozen data {idx['freeze_sha'][:12]}: "
                  + ", ".join(f"{k}={len(v)}" for k, v in self.records.items()))
        return idx

    def _publish_frozen(self, local: Path, extra_idx: dict) -> dict:
        import hashlib
        files = {p.relative_to(local).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                 for p in sorted(local.rglob("*")) if p.is_file()}
        idx = {"files": files, "created_at": utcnow(), **extra_idx,
               "freeze_sha": hashlib.sha256(C.canonical_json(files).encode()).hexdigest()}
        adds = {f"{self.ns}data/{rel}": local / rel for rel in files}
        adds[f"{self.ns}data/SPLITS_SHA256.json"] = json.dumps(idx, indent=2).encode()
        self.store.commit(adds, message="data: freeze manifests + splits")
        log_event("data_frozen", f"published {len(files)} files, freeze {idx['freeze_sha'][:12]}")
        return idx

    def _build_synthetic_data(self, local: Path, M) -> dict:
        from src.pipeline.synthetic import synthetic_records
        recs = synthetic_records(160, seed=0, prefix="fav")
        for r in recs:
            r["dataset"] = "fakeavceleb"
        cross = synthetic_records(48, seed=1, prefix="xds")
        for r in cross:
            r["dataset"] = "synthetic-xds"
        M.write_jsonl(local / "manifests" / "fakeavceleb.jsonl", recs)
        M.write_jsonl(local / "manifests" / "synthetic-xds.jsonl", cross)
        for name, part in (("train", recs[:96]), ("val", recs[96:128]), ("test", recs[128:])):
            M.write_jsonl(local / "splits" / "strict" / f"{name}.jsonl", part)
        return self._publish_frozen(local, {"protocol": {"synthetic": True}})

    def _build_data(self, local: Path, M, S) -> dict:
        cfg = self.cfg
        local.mkdir(parents=True, exist_ok=True)
        wanted = ["fakeavceleb", *cfg["data"]["cross_datasets"]]
        if self.mode == "smoke":
            wanted = [d for d in wanted if d in cfg["smoke"]["datasets"]]
        prov = {"built_at": utcnow(), "git": self.env.get("git"), "corpora": {}}
        records = {}
        for name in wanted:
            root = M.find_kaggle_root(name)
            if root is None:
                if name == "fakeavceleb":
                    raise FileNotFoundError(
                        "FakeAVCeleb is not mounted. Add Input -> Datasets -> "
                        f"{M.KAGGLE_SLUGS[name]} to the notebook.")
                log_event("dataset_missing", f"{name} not mounted -> skipped", logging.WARNING)
                continue
            recs = (M.build_fakeavceleb(root) if name == "fakeavceleb" else
                    M.build_cross_dataset(name, root, cfg["data"]["cross_datasets"][name],
                                          cfg["data"]["subsample_seed"]))
            if self.mode == "smoke" and name != "fakeavceleb":
                recs = S.stratified_subsample(recs, cfg["smoke"]["max_clips_per_split"],
                                              key=lambda r: r["clip_label"])
            records[name] = recs
            prov["corpora"][name] = {"kaggle_slug": M.KAGGLE_SLUGS[name], "root": str(root),
                                     "fingerprint": M.dataset_fingerprint(root),
                                     "summary": M.manifest_summary(recs)}
            log_event("manifest_built", f"{name}: {len(recs)} records", n=len(recs))
        fr, seed = tuple(cfg["data"]["split_fractions"]), cfg["data"]["split_seed"]
        fav = records["fakeavceleb"]
        strict = S.strict_identity_splits(fav, seed, fr)
        legacy = S.legacy_source_splits(fav, seed, fr)
        splits = {"strict": strict["splits"], "legacy": legacy["splits"]}
        for fam, parts in S.video_logo_splits(strict["splits"]).items():
            splits[f"logo_{fam}"] = parts
        if self.mode == "smoke":
            n = cfg["smoke"]["max_clips_per_split"]
            keep = {"strict"}
            splits = {k: {s: S.stratified_subsample(v, n) for s, v in parts.items()}
                      for k, parts in splits.items() if k in keep}
            used = {r["clip_id"] for parts in splits.values() for v in parts.values() for r in v}
            records["fakeavceleb"] = [r for r in fav if r["clip_id"] in used]
        S.assert_no_leakage(splits["strict"])
        audit = {"strict": S.leakage_audit(splits["strict"]),
                 "strict_dropped_clips": len(strict["dropped"])}
        if "legacy" in splits:
            audit["legacy"] = S.leakage_audit(splits["legacy"])
        summary = [dict(r, protocol=k) for k, parts in splits.items()
                   for r in S.split_summary(parts)]
        for name, recs in records.items():
            M.write_jsonl(local / "manifests" / f"{name}.jsonl", recs)
        for proto, parts in splits.items():
            for s, recs in parts.items():
                M.write_jsonl(local / "splits" / proto / f"{s}.jsonl", recs)
        (local / "split_audit.json").write_text(json.dumps(audit, indent=2))
        pd.DataFrame(summary).to_csv(local / "split_summary.csv", index=False)
        (local / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        return self._publish_frozen(local, {"protocol": {
            "split_seed": seed, "fractions": list(fr), "primary": "strict_identity",
            "supplementary": "legacy_source"}})

    # ================================================================ features
    def feature_identity(self) -> str:
        from src.pipeline.encoders import resolve_revision
        from src.pipeline.prepare import feature_set_id
        path = f"{self.ns}data/feature_set.json"
        fs = self.store.read_json(path)
        if fs is None and self.ns:
            fs = self.store.read_json("data/feature_set.json")
            if fs is not None:
                log_event("features_fallback", f"reusing root data/feature_set.json for namespace '{self.ns}'")
        if fs is None:
            f = self.cfg["features"]
            revs = {"video": resolve_revision(f["video_model"]),
                    "audio": resolve_revision(f["audio_model"])}
            fs = {"revisions": revs, "feature_set_id": feature_set_id(self.cfg, revs)
                  + ("-smoke" if self.mode == "smoke" else ""), "created_at": utcnow()}
            self.store.write_json(path, fs, "data: freeze feature-set identity")
        self.revisions, self.fsid = fs["revisions"], fs["feature_set_id"]
        return self.fsid

    def prepare_features(self) -> bool:
        from src.pipeline import manifests as M
        from src.pipeline.prepare import (FeatureExtractor, ensure_yunet, extract_corpus,
                                          write_feature_meta)
        if self.mode == "recovery_test":
            return self._synthetic_features()
        fsid = self.feature_identity()
        if self.data_store.exists(f"features/{fsid}/FEATURES_META.json"):
            log_event("features_ready", f"{fsid} complete on {self.data_store.repo_id}")
            return True
        dev = self.hw["device"]
        extractor = FeatureExtractor(self.cfg, self.revisions, dev)
        yunet = ensure_yunet(self.scratch / "models")
        stats = []
        fav = self.records.get("fakeavceleb", [])
        for corpus, recs in self.records.items():
            fc = self.cfg["features"]
            audio_only = all(r.get("modalities") == "audio" for r in recs)
            shard_n = (min(128, fc["shard_clips"]) if corpus == "fakeavceleb"
                       else (fc["shard_clips_audio"] if audio_only else fc["shard_clips"]))
            n_shards = math.ceil(len(recs) / shard_n)
            prefix = f"features/{fsid}/{corpus}"
            done = {Path(i.path).name for i in self.data_store.list_files(prefix, recursive=False)
                    if i.path.endswith(".json")} if self.data_store is not None else set()
            if len(done) >= n_shards:
                log_event("features_corpus", f"{corpus}: all {n_shards} shards already extracted",
                          corpus=corpus, skipped_shards=n_shards)
                stats.append({"corpus": corpus, "clips": len(recs), "shards": n_shards,
                              "skipped_shards": n_shards, "errors": 0, "face_found": 0,
                              "face_checked": 0, "no_audio": 0, "error_examples": []})
                continue

            root = M.find_kaggle_root(corpus)
            if root is None:
                raise FileNotFoundError(f"{corpus} must be mounted to extract its features "
                                        "(not found in /kaggle/input, /content/drive/MyDrive, etc.)")
            st = extract_corpus(
                corpus=corpus, records=recs, root=str(root), cfg=self.cfg, extractor=extractor,
                data_store=self.data_store, fsid=fsid, scratch=self.scratch,
                multiview_ids={r["clip_id"] for r in fav} if corpus == "fakeavceleb" else set(),
                qacp_ids={r["clip_id"] for r in fav if r["quadrant"] == "RVRA"}
                if corpus == "fakeavceleb" else set(),
                num_workers=0 if corpus in VIDEO_CORPORA else max(1, self.hw["num_workers"]),
                yunet=yunet,
                write_clipcache=corpus in VIDEO_CORPORA,
                batch_size=(8 if corpus in VIDEO_CORPORA else 32) if dev == "cuda" else 4,
                stopwatch=self.stopwatch)
            stats.append(st)
            log_event("features_corpus", f"{corpus}: {st}")
            if st.get("paused"):
                return False
        write_feature_meta(self.data_store, fsid, self.cfg, self.revisions, stats, self.env)
        log_event("features_ready", f"{fsid}: all corpora extracted")
        return True

    def _synthetic_features(self) -> bool:
        from src.pipeline.synthetic import synthetic_qacp_table, synthetic_table
        self.fsid, self.revisions = "synthetic", {}
        d = self.cfg["model"]["d_model"]
        fav = self.records["fakeavceleb"]
        table = synthetic_table(fav, d=d)
        reals = [r for r in fav if r["quadrant"] == "RVRA"]
        self._tables[("proto", "strict")] = {"train": table, "eval": table,
                                             "qacp": synthetic_qacp_table(reals, table)}
        self._tables[("cross", "synthetic-xds")] = synthetic_table(
            self.records["synthetic-xds"], d=d, views=(4,), seed=5)
        log_event("features_ready", "synthetic feature tables (recovery_test mode)")
        return True

    # ================================================================ tables (cached)
    def _store(self):
        from src.pipeline.features import FeatureStore
        return FeatureStore(self.data_store, self.fsid, self.scratch)

    def protocol_tables(self, protocol: str, stage: Optional[str] = None):
        key = ("proto", protocol, "qacp" if stage == "qacp" else "full")
        if ("proto", protocol) in self._tables:
            return self._tables[("proto", protocol)]
        if key not in self._tables:
            # free other protocol tables first (RAM: one protocol at a time)
            for k in [k for k in list(self._tables.keys()) if k[0] == "proto"]:
                del self._tables[k]
            import gc
            gc.collect()
            sp = self.splits[protocol]
            fs = self._store()
            ev = self.cfg["features"]["eval_view_offset"]
            eval_ids = {r["clip_id"] for s in ("val", "test") for r in sp.get(s, [])}
            reals = {r["clip_id"] for s in ("train", "val") for r in sp.get(s, [])
                     if r["quadrant"] == "RVRA"}
            loaders = {
                "eval": lambda: fs.table("fakeavceleb", keep=eval_ids, views={ev}),
                "qacp": lambda: fs.table("fakeavceleb", keep=reals, qacp=True),
            }
            if stage == "qacp":
                # QACP only trains on real clips with eval_view - 96% RAM savings on Colab
                def _load_reals():
                    if "_reals_cache" not in self._tables[key]:
                        self._tables[key]["_reals_cache"] = fs.table("fakeavceleb", keep=reals, views={ev})
                    return self._tables[key]["_reals_cache"]
                loaders["train"] = _load_reals
                loaders["eval"] = _load_reals
            else:
                train_ids = {r["clip_id"] for r in sp["train"]}
                loaders["train"] = lambda: fs.table("fakeavceleb", keep=train_ids)
            self._tables[key] = _LazyTableDict(loaders)
        return self._tables[key]

    @staticmethod
    def _table_protocol(split: str) -> str:
        """LOGO splits are subsets of the strict split -> reuse its (already loaded) tables."""
        return "legacy" if split == "legacy" else "strict"

    def cross_table(self, corpus: str):
        key = ("cross", corpus)
        if key not in self._tables:
            for k in [k for k in list(self._tables.keys()) if k[0] == "cross"]:
                del self._tables[k]
            import gc
            gc.collect()
            self._tables[key] = self._store().table(
                corpus, views={self.cfg["features"]["eval_view_offset"]})
        return self._tables[key]

    # ================================================================ experiments
    def plan(self):
        from src.pipeline.plan import full_plan, select
        pc = self.cfg.get("plan", {})
        plan = select(full_plan(), pc.get("groups"), pc.get("only"))
        if self.mode in ("smoke", "recovery_test"):
            plan = select(full_plan(), None, pc.get("only") or ["qacp", "davidnet", "late-fusion"])
        seeds = self.cfg["seeds"][:1] if self.mode != "full" else self.cfg["seeds"]
        return plan, seeds

    def run_experiments(self) -> dict:
        from src.pipeline.plan import resolve
        from src.pipeline.registry import Registry
        plan, seeds = self.plan()
        done, paused, skipped = [], [], []
        while not self.stopwatch.should_stop():
            made_progress = False
            current_skipped = []
            for seed in seeds:                           # seed-major: a complete seed first
                for spec in plan:
                    if self.stopwatch.should_stop():
                        log_event("session_paused", "time budget reached; stopping here")
                        return {"completed": done, "paused": paused + ["<budget>"], "skipped": current_skipped}
                    if spec.stage == "phase_b" and not self.cfg.get("plan", {}).get("run_phase_b", True):
                        continue
                    cfg_e = resolve(self.cfg, spec, seed)
                    h = C.config_hash(cfg_e)
                    entry = self.registry.register(
                        mode=self.mode, name=spec.name, seed=seed, config_hash=h,
                        meta={"group": spec.group, "stage": spec.stage, "split": spec.split,
                              "init_from": spec.init_from, "description": spec.description})
                    if entry.get("status") == "completed" and entry.get("evaluated"):
                        continue
                    if spec.init_from:
                        dep = self.registry.find(Registry.key(self.mode, spec.init_from, seed))
                        if not dep or dep.get("status") != "completed":
                            current_skipped.append(f"{spec.name}/s{seed} (waiting for {spec.init_from})")
                            continue
                    if not self.registry.claim(entry["exp_id"], self.cfg["session"]["lease_minutes"]):
                        current_skipped.append(f"{spec.name}/s{seed} (claimed elsewhere)")
                        continue
                    try:
                        status = self.run_one(spec, cfg_e, entry)
                    except Exception as e:  # noqa: BLE001 - record, release, continue
                        log_event("error", f"{entry['exp_id']} failed: {type(e).__name__}: {e}",
                                  logging.ERROR)
                        self.registry.set_status(entry["exp_id"], "failed",
                                                 error=f"{type(e).__name__}: {str(e)[:300]}")
                        if self.cfg.get("plan", {}).get("stop_on_error", True):
                            raise
                        continue
                    (done if status == "completed" else paused).append(entry["exp_id"])
                    if status == "paused":
                        return {"completed": done, "paused": paused, "skipped": current_skipped}
                    made_progress = True
            skipped = current_skipped
            if not made_progress:
                break
        return {"completed": done, "paused": paused, "skipped": skipped}

    def _init_state(self, spec, seed: int) -> Optional[dict]:
        from src.pipeline.checkpoint import CheckpointManager
        from src.pipeline.registry import Registry
        if not spec.init_from:
            return None
        dep = self.registry.find(Registry.key(self.mode, spec.init_from, seed))
        if not dep and getattr(self, "ns", None):
            dep = Registry(self.store, "").find(Registry.key(self.mode, spec.init_from, seed))
        if not dep:
            raise RuntimeError(f"{spec.name}: init_from {spec.init_from} not found in registry")
        dep_dir = self.registry.exp_dir(dep) if self.registry.find(Registry.key(self.mode, spec.init_from, seed)) else Registry(self.store, "").exp_dir(dep)
        cm = CheckpointManager(self.store, dep_dir, self.work / "experiments" / Path(dep_dir).name)
        got = cm.load_best()
        if got is None:
            raise RuntimeError(f"{spec.name}: init_from {spec.init_from} has no best_model")
        state, meta = got
        log_event("warm_start_source", f"{spec.name} <- {dep['exp_id']} (epoch {meta.get('epoch')})")
        return state

    def run_one(self, spec, cfg_e: dict, entry: dict) -> str:
        from src.pipeline.checkpoint import CheckpointManager
        from src.pipeline.features import FeatureDataset, QACPFeatureDataset
        from src.pipeline.models import build_baseline, build_davidnet, count_parameters
        from src.pipeline import trainer as T
        seed = int(entry["seed"])
        seed_everything(seed, self.cfg["hardware"]["deterministic"])
        exp_dir = entry["dir"]
        local = self.work / "experiments" / Path(exp_dir).name
        (local / "configs").mkdir(parents=True, exist_ok=True)
        log_event("experiment_started", f"{entry['exp_id']} {spec.name} seed {seed}",
                  exp_id=entry["exp_id"], stage=spec.stage, split=spec.split)
        self._write_configs(local, cfg_e, entry, spec)
        split = self.splits[spec.split]
        tcfg = cfg_e["train"][spec.stage]

        if spec.stage in ("qacp", "stage1", "baseline"):
            tabs = self.protocol_tables(self._table_protocol(spec.split), stage=spec.stage)
        if spec.stage == "qacp":
            reals = lambda s: [r for r in split[s] if r["quadrant"] == "RVRA"]
            train_ds = QACPFeatureDataset(reals("train"), tabs["train"], tabs["qacp"],
                                          tcfg["pseudo_classes"], seed, tcfg["items_per_epoch"])
            val_ds = QACPFeatureDataset(reals("val"), tabs["eval"], tabs["qacp"],
                                        tcfg["pseudo_classes"], seed + 1,
                                        max(256, min(2048, tcfg["items_per_epoch"] // 4)))
            model = build_davidnet(cfg_e, "A")
            job = T.TrainJob(exp=entry, cfg=cfg_e, stage="qacp", model=model, train_ds=train_ds,
                             val_ds=val_ds, objective=T.qacp_objective,
                             validate=T.make_qacp_validator(cfg_e), selection_mode="min",
                             early_stop_fn=T.qacp_floor_stop, num_workers=0)
        elif spec.stage in ("stage1", "baseline"):
            train_ds = FeatureDataset(split["train"], tabs["train"], True, seed)
            val_ds = FeatureDataset(split["val"], tabs["eval"], False, seed)
            if spec.stage == "baseline":
                model = build_baseline(spec.baseline_model, cfg_e["model"]["d_model"])
                obj = T.baseline_objective
            else:
                model = build_davidnet(cfg_e, "A")
                obj = T.stage1_objective
            job = T.TrainJob(exp=entry, cfg=cfg_e, stage=spec.stage, model=model,
                             train_ds=train_ds, val_ds=val_ds, objective=obj,
                             validate=T.make_supervised_validator(obj, cfg_e),
                             sampler_keys=[r[tcfg["sampler_key"]] for r in train_ds.records],
                             init_state=self._init_state(spec, seed), num_workers=0)
        elif spec.stage == "phase_b":
            job = self._phase_b_job(spec, cfg_e, entry, split, seed)
        elif spec.stage == "external":
            job = self._external_job(spec, cfg_e, entry, split, seed, local)
        else:
            raise KeyError(spec.stage)

        n_params = count_parameters(job.model)
        log_event("model_initialized", f"{spec.name}: {n_params['total'] / 1e6:.1f} M params "
                  f"({n_params['trainable'] / 1e6:.1f} M trainable)", **n_params)
        ckpt = CheckpointManager(self.store, exp_dir, local, cfg_e["checkpoint"]["keep_last"],
                                 cfg_e["checkpoint"]["background_upload"], C.config_hash(cfg_e))
        ctx = T.RunContext(hw=self.hw, ckpt=ckpt, local_dir=local, registry=self.registry,
                           stopwatch=self.stopwatch,
                           heartbeat_minutes=cfg_e["session"]["heartbeat_minutes"],
                           extra_files={"logs/pipeline.jsonl": self.log_path})
        result = T.train(job, ctx)
        if result["status"] == "paused":
            self.registry.set_status(entry["exp_id"], "paused", progress_step=result["steps"])
            return "paused"
        result["parameters"] = n_params
        self._evaluate_and_publish(spec, cfg_e, entry, job, ckpt, local, result)
        return "completed"

    def _phase_b_job(self, spec, cfg_e, entry, split, seed):
        from src.pipeline import trainer as T
        from src.pipeline.models import build_davidnet
        from src.pipeline.phase_b import ClipCacheIndex, PixelDataset, phase_b_param_groups
        t = cfg_e["train"]["phase_b"]
        cache = ClipCacheIndex(self.data_store, "fakeavceleb", self.scratch)
        model = build_davidnet(cfg_e, "B", self.revisions, t["unfreeze_top_blocks_video"],
                               t["unfreeze_top_blocks_audio"], t["gradient_checkpointing"])
        train_ds = PixelDataset(split["train"], cache, t["temporal_augment"], seed)
        val_ds = PixelDataset(split["val"], cache, False, seed)
        obj = lambda m, b, c, tr: T.stage1_objective(m, b, c, tr, stage="phase_b")
        return T.TrainJob(exp=entry, cfg=cfg_e, stage="phase_b", model=model, train_ds=train_ds,
                          val_ds=val_ds, objective=obj,
                          validate=T.make_supervised_validator(obj, cfg_e),
                          sampler_keys=[r[t["sampler_key"]] for r in train_ds.records],
                          param_groups=phase_b_param_groups(model, cfg_e),
                          init_state=self._init_state(spec, seed), bytes_per_sample_gb=0.9,
                          num_workers=0)   # per-epoch worker processes get the container killed

    def _external_job(self, spec, cfg_e, entry, split, seed, local):
        from src.pipeline import trainer as T
        from src.pipeline.external import build_external
        from src.pipeline.phase_b import ClipCacheIndex, PixelDataset
        t = cfg_e["train"]["external"]
        cache = ClipCacheIndex(self.data_store, "fakeavceleb", self.scratch)
        model = build_external(spec.baseline_model, cfg_e, self.scratch / "models")
        (local / "configs").mkdir(parents=True, exist_ok=True)
        (local / "configs" / "external_provenance.json").write_text(
            json.dumps(model.provenance, indent=2, default=str))
        train_ds = PixelDataset(split["train"], cache, True, seed)
        val_ds = PixelDataset(split["val"], cache, False, seed)
        return T.TrainJob(exp=entry, cfg=cfg_e, stage="external", model=model,
                          train_ds=train_ds, val_ds=val_ds, objective=T.baseline_objective,
                          validate=T.make_supervised_validator(T.baseline_objective, cfg_e),
                          sampler_keys=[r[t["sampler_key"]] for r in train_ds.records],
                          bytes_per_sample_gb=0.35 if spec.baseline_model == "effnet_b4" else 0.05,
                          num_workers=0)

    # ================================================================ evaluation + upload
    def _write_configs(self, local: Path, cfg_e: dict, entry: dict, spec):
        c = local / "configs"
        (c / "config.json").write_text(json.dumps(cfg_e, indent=2, default=str))
        (c / "environment.json").write_text(json.dumps(dict(self.env, hardware=self.hw),
                                                       indent=2, default=str))
        if not hasattr(self, "_freeze_txt"):
            self._freeze_txt = pip_freeze()
        (c / "requirements.lock.txt").write_text(self._freeze_txt)
        meta = {"experiment_id": entry["exp_id"], "name": spec.name, "seed": entry["seed"],
                "group": spec.group, "stage": spec.stage, "split_protocol": spec.split,
                "init_from": spec.init_from, "description": spec.description,
                "components": spec.components, "config_hash": entry["config_hash"],
                "feature_set_id": self.fsid, "model_revisions": self.revisions,
                "data_freeze_sha": self._freeze, "git": self.env.get("git"),
                "hf_repo": self.store.repo_id, "hf_revision_at_session_start": self.hub_revision_at_start,
                "hf_data_repo": self.data_store.repo_id, "registered_at": entry.get("created_at"),
                "mode": self.mode}
        (c / "experiment_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    def _loader(self, ds, stage: Optional[str] = None):
        from torch.utils.data import DataLoader
        from src.pipeline.features import FeatureDataset, collate
        bs = 16 if stage in ("phase_b", "external") else (64 if self.hw["device"] == "cuda" else 16)
        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=0,   # no worker processes: see _DecodePool._start
                          collate_fn=collate,
                          pin_memory=self.hw["pin_memory"])

    def _evaluate_and_publish(self, spec, cfg_e, entry, job, ckpt, local: Path, result: dict):
        from src.pipeline import evaluation as E
        from src.pipeline import reporting as R
        from src.pipeline.features import FeatureDataset
        exp_dir = entry["dir"]
        metrics = local / "metrics"
        preds = local / "predictions"
        metrics.mkdir(parents=True, exist_ok=True)
        preds.mkdir(parents=True, exist_ok=True)
        summary = {"training": result}
        if spec.stage != "qacp":
            if self.store.exists(f"{exp_dir}/metrics/test_metrics.json"):
                log_event("test_already_evaluated", f"{entry['exp_id']}: kept existing test "
                          "metrics (the test split is evaluated exactly once per experiment)")
            else:
                self._evaluate(spec, cfg_e, entry, job, ckpt, local, summary, E)
        R.experiment_figures(local)
        log_event("figures_generated", f"{entry['exp_id']}")
        (metrics / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (local / "README.md").write_text(self._experiment_card(spec, entry, summary))
        adds = {}
        for sub in ("configs", "metrics", "predictions", "figures", "logs"):
            for p in (local / sub).rglob("*"):
                if p.is_file():
                    adds[f"{exp_dir}/{sub}/{p.relative_to(local / sub).as_posix()}"] = p
        adds[f"{exp_dir}/README.md"] = local / "README.md"
        drop = (ckpt.resume_state_delete_ops()
                if cfg_e["checkpoint"]["prune_resume_state_on_completion"] else [])
        self.store.commit(adds, message=f"{entry['exp_id']}: results", delete_folders=drop)
        log_event("results_uploaded", f"{entry['exp_id']}: {len(adds)} files")
        test = summary.get("test", {})
        self.registry.set_status(entry["exp_id"], "completed", evaluated=True,
                                 best_val=result["best"], train_hours=result["train_hours"],
                                 test_clip_auc=(test.get("clip") or {}).get("auc"),
                                 completed_at=utcnow())
        log_event("experiment_completed", f"{entry['exp_id']} {spec.name} seed {entry['seed']}")

    def _evaluate(self, spec, cfg_e, entry, job, ckpt, local, summary, E):
        from src.pipeline.features import FeatureDataset
        best, meta = ckpt.load_best()
        job.model.load_state_dict(best)
        model = job.model.to(self.hw["device"])
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.hw["precision"])
        version = f"{entry['exp_id']}@epoch{meta['epoch']}:{meta['sha256'][:12]}"
        split = self.splits[spec.split]
        seed = int(entry["seed"])
        if spec.stage in ("phase_b", "external"):
            from src.pipeline.phase_b import ClipCacheIndex, PixelDataset
            cache = ClipCacheIndex(self.data_store, "fakeavceleb", self.scratch)
            mk = lambda recs: PixelDataset(recs, cache, False, seed)
        else:
            tabs = self.protocol_tables(self._table_protocol(spec.split), stage="eval")
            mk = lambda recs: FeatureDataset(recs, tabs["eval"], False, seed)
        dev = self.hw["device"]
        pv = E.predict(model, self._loader(mk(split["val"]), spec.stage), dev, amp, entry["exp_id"], version, "val")
        thr = E.fit_thresholds(pv, self.cfg["evaluation"]["threshold_policy"])
        val_m = E.evaluate_frame(pv, thr, self.cfg["evaluation"], seed)
        pt = E.predict(model, self._loader(mk(split["test"]), spec.stage), dev, amp, entry["exp_id"], version, "test")
        test_m = E.evaluate_frame(pt, thr, self.cfg["evaluation"], seed)
        test_m["evaluated_checkpoint"] = version
        pv.to_csv(local / "predictions" / "val.csv", index=False)
        pt.to_csv(local / "predictions" / "test.csv", index=False)
        m = local / "metrics"
        (m / "thresholds.json").write_text(json.dumps({"policy": self.cfg["evaluation"]["threshold_policy"],
                                                       "fitted_on": "validation", **thr}, indent=2))
        (m / "validation_metrics.json").write_text(json.dumps(val_m, indent=2, default=str))
        (m / "test_metrics.json").write_text(json.dumps(test_m, indent=2, default=str))
        log_event("evaluation_completed", f"{entry['exp_id']} test: clip AUC "
                  f"{(test_m.get('clip') or {}).get('auc', float('nan')):.4f}")
        self._flat_tables(pt, test_m, thr, m)
        cross = {}
        if spec.evaluate_cross and spec.stage != "qacp":
            for corpus in [c for c in self.records if c != "fakeavceleb"]:
                recs = self.records[corpus]
                if spec.stage in ("phase_b", "external"):
                    from src.pipeline.phase_b import ClipCacheIndex, PixelDataset
                    from src.pipeline.manifests import find_kaggle_root
                    audio_only = all(r.get("modalities") == "audio" for r in recs)
                    cc = None if audio_only else ClipCacheIndex(self.data_store, corpus, self.scratch)
                    ds = PixelDataset(recs, cc, False, seed, media_root=str(find_kaggle_root(corpus)),
                                      dcfg=self.cfg["data"])
                else:
                    ds = FeatureDataset(recs, self.cross_table(corpus), False, seed)
                pc = E.predict(model, self._loader(ds, spec.stage), dev, amp, entry["exp_id"], version,
                               f"cross:{corpus}")
                pc.to_csv(local / "predictions" / f"cross_{corpus}.csv", index=False)
                cross[corpus] = E.evaluate_frame(pc, thr, self.cfg["evaluation"], seed)
                cross[corpus]["per_generator"] = E.group_metrics(pc, "clip", "generator", thr["clip"])
            (m / "cross_dataset_metrics.json").write_text(json.dumps(cross, indent=2, default=str))
        summary.update(validation=val_m, test=test_m, cross=cross, thresholds=thr)

    @staticmethod
    def _flat_tables(pt: pd.DataFrame, test_m: dict, thr: dict, m: Path):
        from src.pipeline import evaluation as E
        rows = []
        for task in ("video", "audio", "clip"):
            if task in test_m:
                rows.append({"task": task, **{k: v for k, v in test_m[task].items()
                                              if not isinstance(v, list)}})
        pd.DataFrame(rows).to_csv(m / "test_metrics.csv", index=False)
        q = test_m.get("quadrant")
        if q:
            pc = pd.DataFrame([{"class": k, **v} for k, v in q["per_class"].items()])
            pc.to_csv(m / "per_class_metrics.csv", index=False)
            pd.concat([pc, pd.DataFrame([{"class": "macro avg", "f1": q["macro_f1"]},
                                         {"class": "weighted avg", "f1": q["weighted_f1"]},
                                         {"class": "accuracy", "f1": q["accuracy"]}])]
                      ).to_csv(m / "classification_report.csv", index=False)
            pd.DataFrame(q["confusion"], index=[r["class"] for r in pc.to_dict("records")]
                         ).to_csv(m / "confusion_matrix.csv")
        pd.DataFrame(E.group_metrics(pt, "clip", "generator", thr["clip"])).to_csv(
            m / "per_generator.csv", index=False)
        fair = []
        for col in ("race", "gender"):
            for task in ("video", "audio"):
                for r in E.group_metrics(pt, task, col, thr[task]):
                    fair.append({"attribute": col, "task": task, **r})
        pd.DataFrame(fair).to_csv(m / "fairness.csv", index=False)

    def _experiment_card(self, spec, entry, summary) -> str:
        t = summary.get("test", {})
        line = lambda k: (f"| {k} | {t[k]['auc']:.4f} | {t[k]['eer']:.4f} | {t[k]['f1']:.4f} |"
                          if k in t and t[k].get("auc_defined") else f"| {k} | -- | -- | -- |")
        tr = summary["training"]
        return "\n".join([
            f"# {entry['exp_id']} - {spec.name} (seed {entry['seed']})", "",
            spec.description, "",
            f"- group: `{spec.group}`, stage: `{spec.stage}`, split protocol: `{spec.split}`",
            f"- initialised from: `{spec.init_from or 'none'}`",
            f"- config hash: `{entry['config_hash']}`, feature set: `{self.fsid}`",
            f"- best epoch {tr['best']['epoch']} ({tr['best']['metric']}), "
            f"{tr['epochs_run']} epochs, {tr['train_hours']} h, stop: {tr['stop_reason']}", "",
            "| test task | AUC | EER | F1@val-thr |", "|---|---|---|---|",
            line("video"), line("audio"), line("clip"), "",
            "Files: `configs/` (config, environment, metadata, requirements), `metrics/`, "
            "`predictions/`, `figures/`, `logs/`, `best_model/`.", ""])

    # ================================================================ robustness
    ROBUSTNESS_TARGETS = ("davidnet", "davidnet-e2e", "late-fusion", "video-probe", "audio-probe",
                          "aasist", "effnet-b4")

    def _model_for(self, cfg_e: dict):
        from src.pipeline.models import build_baseline, build_davidnet
        ex = cfg_e["experiment"]
        if ex["stage"] == "baseline":
            return build_baseline(ex["baseline_model"], cfg_e["model"]["d_model"])
        if ex["stage"] == "phase_b":
            return build_davidnet(cfg_e, "B", self.revisions)
        if ex["stage"] == "external":
            from src.pipeline.external import build_external
            return build_external(ex["baseline_model"], cfg_e, self.scratch / "models")
        return build_davidnet(cfg_e, "A")

    def run_robustness(self) -> dict:
        """Test-set robustness of every completed target model (strict split only).

        Loops perturbation-first so each degraded feature set is computed once and shared
        by all Phase-A models/seeds, then freed. Idempotent: experiments that already have
        metrics/robustness.json are skipped."""
        from src.pipeline import evaluation as E
        from src.pipeline.checkpoint import CheckpointManager
        from src.pipeline.features import FeatureDataset
        from src.pipeline.phase_b import ClipCacheIndex
        from src.pipeline.robustness import (PerturbedPixelDataset, perturbations,
                                             perturbed_feature_table)
        if self.mode == "recovery_test":
            return {}
        reg = self.registry.load()
        todo = []
        for e in reg["experiments"].values():
            if (e.get("mode") == self.mode and e.get("status") == "completed"
                    and e["name"] in self.ROBUSTNESS_TARGETS and e.get("split") == "strict"):
                d = self.registry.exp_dir(e)
                if not self.store.exists(f"{d}/metrics/robustness.json"):
                    todo.append((e, d))
        if not todo:
            log_event("robustness_skipped", "nothing to do")
            return {}
        if self.fsid is None:
            self.feature_identity()
        cache = ClipCacheIndex(self.data_store, "fakeavceleb", self.scratch)
        test = self.splits["strict"]["test"]
        models = []
        for e, d in todo:
            cfg_e = self.store.read_json(f"{d}/configs/config.json")
            thr = self.store.read_json(f"{d}/metrics/thresholds.json") or {}
            model = self._model_for(cfg_e)
            got = CheckpointManager(self.store, d, self.work / "experiments" / Path(d).name).load_best()
            if got is None:
                continue
            model.load_state_dict(got[0])
            models.append({"entry": e, "dir": d, "cfg": cfg_e, "model": model,
                           "thr": {k: thr.get(k, 0.5) for k in ("video", "audio", "clip")},
                           "phase_b": cfg_e["experiment"]["stage"] in ("phase_b", "external"),
                           "results": {}})
        need_features = any(not m["phase_b"] for m in models)
        extractor = None
        if need_features:
            from src.pipeline.prepare import FeatureExtractor
            extractor = FeatureExtractor(self.cfg, self.revisions, self.hw["device"])
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.hw["precision"])
        ev = dict(self.cfg["evaluation"], bootstrap=0)
        codecs = {}
        for name, spec in perturbations().items():
            if self.stopwatch.should_stop():
                log_event("session_paused", "time budget reached during robustness sweep")
                return {"paused": True}
            table = None
            if need_features:
                table, used = perturbed_feature_table(test, cache, name, spec, extractor)
                codecs[name] = used
            pix = PerturbedPixelDataset(test, cache, name, spec) if any(m["phase_b"] for m in models) else None
            for m in models:
                ds = pix if m["phase_b"] else FeatureDataset(test, table, False, int(m["entry"]["seed"]))
                mdl = m["model"].to(self.hw["device"])
                df = E.predict(mdl, self._loader(ds), self.hw["device"], amp)
                met = E.evaluate_frame(df, m["thr"], ev)
                m["results"][name] = {"video_auc": (met.get("video") or {}).get("auc"),
                                      "audio_auc": (met.get("audio") or {}).get("auc"),
                                      "clip_auc": (met.get("clip") or {}).get("auc"),
                                      "video_eer": (met.get("video") or {}).get("eer"),
                                      "audio_eer": (met.get("audio") or {}).get("eer"),
                                      "n": met.get("n")}
                mdl.to("cpu")
            log_event("robustness_condition", f"{name}: {len(models)} models evaluated")
            del table, pix
            if self.hw["device"] == "cuda":
                torch.cuda.empty_cache()
        for m in models:
            out = {"experiment_id": m["entry"]["exp_id"], "name": m["entry"]["name"],
                   "seed": m["entry"]["seed"], "thresholds": m["thr"], "codecs": codecs,
                   "reference": "clean = test windows re-decoded from the clip cache "
                                "(JPEG q87 frames); compare degradations against it",
                   "per_perturbation": m["results"]}
            rows = [{"perturbation": k, **v} for k, v in m["results"].items()]
            local = self.work / "experiments" / Path(m["dir"]).name / "metrics"
            local.mkdir(parents=True, exist_ok=True)
            (local / "robustness.json").write_text(json.dumps(out, indent=2, default=str))
            pd.DataFrame(rows).to_csv(local / "robustness.csv", index=False)
            self.store.commit({f"{m['dir']}/metrics/robustness.json": local / "robustness.json",
                               f"{m['dir']}/metrics/robustness.csv": local / "robustness.csv"},
                              message=f"{m['entry']['exp_id']}: robustness sweep")
        log_event("robustness_completed", f"{len(models)} experiments")
        return {"evaluated": [m["entry"]["exp_id"] for m in models]}

    # ================================================================ reports
    def collect_results(self) -> list[dict]:
        reg = self.registry.load()
        out = []
        tmp = self.work / "_collect"
        for e in reg["experiments"].values():
            if e.get("status") != "completed" or e.get("mode") != self.mode or e.get("stage") == "qacp":
                continue
            d = self.registry.exp_dir(e)
            try:
                test = self.store.read_json(f"{d}/metrics/test_metrics.json")
                cross = self.store.read_json(f"{d}/metrics/cross_dataset_metrics.json") or {}
                cfg = self.store.read_json(f"{d}/configs/config.json")
                p = self.store.download(f"{d}/predictions/test.csv", tmp)
                preds = pd.read_csv(p)
            except Exception as ex:  # noqa: BLE001
                log_event("collect_skip", f"{e['exp_id']}: {ex}", logging.WARNING)
                continue
            rob = (self.store.read_json(f"{d}/metrics/robustness.json")
                   if e["name"] in self.ROBUSTNESS_TARGETS else None)
            out.append({"robustness": rob, "exp_id": e["exp_id"], "name": e["name"], "seed": e["seed"],
                        "group": e.get("group"), "split": e.get("split"), "test": test or {},
                        "cross": cross, "components": cfg["experiment"]["components"],
                        "preds_test": preds, "config": cfg})
        return out

    def build_reports(self) -> dict:
        from src.pipeline import evaluation as E
        from src.pipeline import reporting as R
        results = self.collect_results()
        if not results:
            log_event("reports_skipped", "no completed experiments yet")
            return {}
        level = self.cfg["evaluation"]["ci_level"]
        by = R.aggregate(results, self.cfg["evaluation"])
        out = self.work / "reports"
        tabs, figs = out / "tables", out / "figures"
        files = []
        main_names = [n for n in ("video-probe", "audio-probe", "late-fusion", "aasist", "effnet-b4",
                                  "davidnet", "davidnet-e2e") if n in by]
        ablations = [n for n, rs in by.items() if rs[0]["group"] == "ablation"]
        corpora = sorted({c for rs in by.values() for r in rs for c in (r.get("cross") or {})})
        delong = self._delong(by, "davidnet", ablations + [n for n in main_names if n != "davidnet"])
        files += R.write_table(R.main_table(by, main_names, level), tabs, "main_results",
                               "In-domain test results (strict identity-disjoint FakeAVCeleb split); "
                               "mean $\\pm$ std over seeds.")
        files += R.write_table(R.ablation_table(by, "davidnet", ablations, level, delong["counts"]),
                               tabs, "ablation_results",
                               "Ablations. Component columns: QACP, sync, disentanglement, "
                               "localization, multi-task, modality dropout, MISMATCH class, "
                               "copy-synthesis, self-blending. Significance: DeLong test vs. the "
                               "full model per seed, Holm-corrected across ablations.")
        if corpora:
            files += R.write_table(R.cross_table(by, main_names, corpora, level), tabs,
                                   "cross_dataset_results",
                                   "Zero-shot cross-dataset results (no training or threshold "
                                   "fitting on these corpora). DR = detection rate (fakes-only corpora).")
            files += R.fig_cross(by, main_names, corpora, figs, level)
        if "davidnet" in by:
            files += R.write_table(R.per_class_table(by, "davidnet", level), tabs,
                                   "per_class_results", "Per-quadrant test results of DAVID-Net.")
            files += R.write_table(R.hyperparameter_table(by["davidnet"][0]["config"]), tabs,
                                   "hyperparameters", "Hyperparameters of the proposed model.")
        logo = [n for n, rs in by.items() if rs[0]["group"] == "logo"]
        if logo:
            files += R.write_table(R.main_table(by, sorted(logo), level), tabs, "logo_results",
                                   "Leave-one-generator-family-out (video families).")
        if "davidnet-legacy" in by:
            files += R.write_table(R.main_table(by, ["davidnet", "davidnet-legacy"], level), tabs,
                                   "protocol_comparison",
                                   "Strict vs. legacy (source-identity) split: the legacy split "
                                   "shares identities between train and test.")
        rob = [{"name": r["name"], "seed": r["seed"],
                "per_perturbation": r["robustness"]["per_perturbation"]}
               for r in results if r.get("robustness")]
        if rob:
            from src.pipeline.robustness import fig_robustness, robustness_rows
            rrows = robustness_rows(rob)
            rdf = pd.DataFrame(rrows)
            fmt = rdf.copy()
            for c in ("Video AUC", "Audio AUC", "Clip AUC"):
                fmt[c] = [("--" if pd.isna(m) else
                           (f"{m:.3f}" if pd.isna(sd) else f"{m:.3f} $\\pm$ {sd:.3f}"))
                          for m, sd in zip(rdf[c], rdf[f"{c} std"])]
            fmt = fmt[["Model", "Perturbation", "Family", "Seeds", "Video AUC", "Audio AUC", "Clip AUC"]]
            files += R.write_table(fmt, tabs, "robustness_results",
                                   "Test AUC under degradations (thresholds from clean validation; "
                                   "mean $\\pm$ std over seeds).")
            files += fig_robustness(rrows, figs, main_names)
        files += R.fig_ablation(by, "davidnet", ablations, figs, level)
        files += R.fig_pooled_roc(by, main_names, figs)
        audit = self.work / "data" / "split_audit.json"
        if audit.exists():
            a = json.loads(audit.read_text())
            rows = [{"Protocol": p, "Pair": k, **v} for p in ("strict", "legacy") if p in a
                    for k, v in a[p].items()]
            files += R.write_table(pd.DataFrame(rows), tabs, "split_leakage_audit",
                                   "Identity leakage audit of the split protocols.")
        summary = {"generated_at": utcnow(), "experiments": len(results),
                   "models": {n: len(rs) for n, rs in by.items()}, "delong": delong}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        raw = pd.DataFrame([{"exp_id": r["exp_id"], "name": r["name"], "seed": r["seed"],
                             "group": r["group"], **{f"{t}_{k}": (r["test"].get(t) or {}).get(k)
                                                     for t in ("video", "audio", "clip")
                                                     for k in ("auc", "eer", "f1", "pr_auc")}}
                            for r in results])
        raw.to_csv(out / "raw_results.csv", index=False)
        adds = {f"{self.ns}reports/{p.relative_to(out).as_posix()}": p
                for p in out.rglob("*") if p.is_file()}
        self.store.commit(adds, message="reports: aggregate tables + figures")
        log_event("results_uploaded", f"reports: {len(adds)} files")
        return summary

    @staticmethod
    def _delong(by: dict, ref: str, others: list[str]) -> dict:
        from src.pipeline import evaluation as E
        from src.pipeline.evaluation import task_frame
        per_seed: dict = {}
        for n in others:
            for r in by.get(n, []):
                refs = [x for x in by.get(ref, []) if x["seed"] == r["seed"]]
                if not refs:
                    continue
                a = refs[0]["preds_test"].set_index("sample_id")
                b = r["preds_test"].set_index("sample_id")
                common = a.index.intersection(b.index)
                if not len(common):
                    continue
                ya, pa = task_frame(a.loc[common].reset_index(), "clip")
                _, pb = task_frame(b.loc[common].reset_index(), "clip")
                if len(ya) != len(pb):
                    continue
                per_seed.setdefault(r["seed"], {})[n] = E.delong_test(ya, pa, pb)
        adjusted, counts = {}, {}
        for seed, tests in per_seed.items():
            adj = E.holm_bonferroni({n: t["p_value"] for n, t in tests.items()})
            for n, p in adj.items():
                adjusted.setdefault(n, {})[seed] = {"p_holm": p, **tests[n]}
        for n, seeds in adjusted.items():
            k = sum(1 for v in seeds.values() if v["p_holm"] < 0.05)
            counts[n] = f"{k}/{len(seeds)}"
        return {"tests": adjusted, "counts": counts}

    # ================================================================ session record
    def publish_session_record(self, extra: dict | None = None) -> str:
        """sessions/<stamp>/: config, environment, pip freeze, session log -- one per
        Kaggle session, so every run of the notebook is auditable after the fact."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        d = f"{self.ns}sessions/{stamp}"
        if not hasattr(self, "_freeze_txt"):
            self._freeze_txt = pip_freeze()
        adds = {f"{d}/config.json": json.dumps(self.cfg, indent=2, default=str).encode(),
                f"{d}/environment.json": json.dumps(dict(self.env, hardware=self.hw),
                                                    indent=2, default=str).encode(),
                f"{d}/requirements.lock.txt": self._freeze_txt.encode(),
                f"{d}/session.json": json.dumps({"elapsed_hours": self.stopwatch.elapsed_h(),
                                                 "feature_set_id": self.fsid,
                                                 "data_freeze_sha": self._freeze,
                                                 **(extra or {})}, indent=2, default=str).encode()}
        if self.log_path.exists():
            adds[f"{d}/pipeline.jsonl"] = self.log_path
        self.store.commit(adds, message=f"session record {stamp}")
        return d

    # ================================================================ final report
    def final_report(self) -> str:
        reg = self.registry.load()
        rows = [e for e in reg["experiments"].values() if e.get("mode") == self.mode]
        by_status = {}
        for e in rows:
            by_status.setdefault(e["status"], []).append(e)
        lines = [f"# {self.cfg['project']['name']} - experiment repository ({self.mode})", "",
                 f"Updated {utcnow()} by `{self.hw['gpu_name']}` session "
                 f"({self.stopwatch.elapsed_h():.2f} h).", "",
                 "| status | experiments |", "|---|---|"]
        for s, es in sorted(by_status.items()):
            lines.append(f"| {s} | {len(es)} |")
        lines += ["", "| ID | name | seed | status | test clip AUC |", "|---|---|---|---|---|"]
        for e in sorted(rows, key=lambda e: e["exp_id"]):
            auc = e.get("test_clip_auc")
            lines.append(f"| {e['exp_id']} | {e['name']} | {e['seed']} | {e['status']} | "
                         f"{'' if auc is None else f'{auc:.4f}'} |")
        lines += ["", "Layout: `data/` frozen manifests + splits (SPLITS_SHA256.json), "
                  "`registry/experiments.json`, `experiments/EXP_###_<name>_s<seed>/` "
                  "(`checkpoints/`, `best_model/`, `configs/`, `metrics/`, `predictions/`, "
                  "`figures/`, `logs/`), `reports/` (manuscript tables + figures).", ""]
        text = "\n".join(lines)
        path = f"{self.ns}README.md" if self.ns else "README.md"
        try:
            self.store.commit({path: text.encode()}, message="README: status")
        except Exception as e:  # noqa: BLE001
            log_event("readme_failed", str(e)[:200], logging.WARNING)
        return text
