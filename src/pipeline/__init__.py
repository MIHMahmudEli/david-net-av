"""Q1 experiment pipeline: Kaggle-trained, Hugging Face-persisted, resumable.

Modules
-------
env         logging, seeding, RNG capture/restore, environment + GPU auto-config
config      the research CONFIG schema, validation, and a stable config hash
hub         Hugging Face store: verified, retrying, atomic commits
registry    experiment IDs (EXP_###), claims, never-overwrite guarantees
checkpoint  full training state -> local -> HF, discovery, verified resume
splits      strict identity-disjoint FakeAVCeleb splits + leakage audit + LOGO
manifests   corrected test-only cross-dataset manifests
prepare     decode -> face crop -> frozen SSL features (+ Phase-B clip cache)
features    in-memory feature store + datasets for Phase A
trainer     one training loop for QACP / Stage 1 / baselines, Phase A and B
evaluation  predictions, thresholds fitted on validation, metrics, DeLong
reporting   manuscript figures (PNG+PDF) and tables (CSV+LaTeX)
plan        the experiment matrix (baseline -> proposed -> ablations -> final)
driver      the resumable session driver the Kaggle notebook calls
"""
