"""Dataset configuration: Kaggle dataset slugs and storage paths.

This module is the single canonical configuration file for external dataset
owner slugs used in data loading across the repository.
"""
from __future__ import annotations

import os

# Authoritative Kaggle dataset owner slug for FakeAVCeleb
FAKEAVCELEB_DATASET_SLUG: str = os.environ.get(
    "FAKEAVCELEB_DATASET_SLUG",
    "aicontentdetections/fakeavceleb-v1-2",
)

# Ordered mapping of dataset short names to their canonical Kaggle dataset slugs
KAGGLE_SLUGS: dict[str, str] = {
    "fakeavceleb": FAKEAVCELEB_DATASET_SLUG,
    "celeb-df-v2": "reubensuju/celeb-df-v2",
    "dfdc-10": "pranay22077/dfdc-10",
    "deepfaketimit": "fahimaislam1812/deepfaketimit",
    "asvspoof2019-la": "anishsarkar22/asvpoof-2019-dataset-la",
    "in-the-wild": "abdallamohamed312/in-the-wild-audio-deepfake",
    "wavefake": "walimuhammadahmad/fakeaudio",
}
