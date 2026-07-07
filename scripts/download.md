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

## After download
1. Put raw data under `data/<dataset>/` (git-ignored).
2. Run the per-dataset converter to produce a unified manifest:
   `python scripts/build_manifest.py --dataset fakeavceleb --root data/fakeavceleb --out src/data/manifests/fakeavceleb.jsonl`
   (Write one small converter per dataset — they only need to emit the schema in `docs/03_datasets.md`.)
3. Preprocess into shards: `python -m src.data.preprocess --manifest ... --out data/shards/<dataset>`
4. Build split files (`src/data/splits/*.jsonl`) — subject-disjoint; commit them for reproducibility.
