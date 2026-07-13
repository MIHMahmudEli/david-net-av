# Dataset Access (request these FIRST — approvals are slow)

None of these are redistributed here. Each requires accepting a EULA / filling a form.
Record the license + your approval date in this file as you go.

| Dataset | Access | Notes |
|---------|--------|-------|
| FakeAVCeleb | Request form (Google form from the authors' GitHub) | Native 4-quadrant labels — primary training set |
| AV-Deepfake1M / 1M++ | Authors' GitHub + form / challenge page | Temporal localization; large scale |
| LAV-DF | Authors' GitHub request | Localized content-driven forgeries |
| DFDC | Meta / Kaggle DFDC page | Cross-dataset test |
| KoDF | Request form (Korean DeepFake) | Cross-demographic test |
| DeepfakeTIMIT | Idiap request | Classic face-swap |
| Celeb-DF v2 | Request form | Video-branch cross-dataset |
| ASVspoof 2019/2021 | Edinburgh DataShare (open) | Audio anti-spoofing |
| In-the-Wild (voice) | Public download | Real-world voice-clone generalization |
| WaveFake / LibriSeVoc | Public | Extra vocoder/TTS diversity |

## After download (FakeAVCeleb — the full chain)
1. Put raw data under `data/fakeavceleb/` (git-ignored).
2. Manifest + subject-disjoint + LOGO splits in one command:
   ```
   python scripts/build_manifest.py --root data/fakeavceleb \
       --out src/data/manifests/fakeavceleb.jsonl \
       --splits-dir src/data/splits/fakeavceleb --seed 42
   ```
   Commit the generated split files for reproducibility.
3. Preprocess into shards (face/mouth crops + audio):
   `python -m src.data.preprocess --manifest src/data/manifests/fakeavceleb.jsonl --raw-root data/fakeavceleb --out data/shards/fakeavceleb`
4. Cache SSL features (Phase A):
   `python -m src.data.extract_features --config configs/david_net.yaml --manifest src/data/manifests/fakeavceleb.jsonl --out data/feats/fakeavceleb`
5. Train: `pretrain_qacp` (Stage 0) then `train` (Stage 1) — see README quickstart.

(Other datasets: write one small converter each in `scripts/`, emitting the schema
in `docs/03_datasets.md` §4 — the rest of the chain is dataset-agnostic.)
