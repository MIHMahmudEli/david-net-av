# Journal Shortlist (Scopus-indexed) and Submission Strategy

> Prepared July 2026 at the supervisor's request: identify Scopus-indexed
> journals aligned with the scope of this research, target manuscript length
> 50–60 pages. **Verify each journal's current quartile, impact factor, APC,
> and scope statement on Scopus / JCR / the journal homepage at submission
> time** — these change year to year.

## 1. Lessons from the IPM desk rejection (July 2026)

The Information Processing & Management editor desk-rejected the earlier
(text-focused PIRD) manuscript with three explicit reasons that apply to any
future submission from this group:

1. **Stale references** — "reference the most updated articles from the
   current year." → This manuscript now carries ~40 references from
   2022–2025; before submission, add 2–4 citations from **2026 issues of the
   target journal itself** (search the journal's "articles in press" for
   deepfake / multimodal forensics papers and cite the relevant ones).
2. **Weak / outdated baselines** — "leveraging state-of-the-art baselines."
   → The evaluation now includes AVFF (CVPR 2024), UMMAFormer, BA-TFD+, and
   ASVspoof-era SSL audio baselines. Do not cut these to save compute.
3. **Scope fit** — IPM is an information-science venue; a detection-systems
   paper reads as out of scope there. Choose venues whose recent issues
   actually publish deepfake-detection papers (all venues below do).

## 2. Tier 1 — primary targets (Q1, strong scope fit)

| Journal | Publisher | Why it fits | Risk |
|---|---|---|---|
| **IEEE Trans. on Information Forensics and Security (TIFS)** | IEEE | The natural home: AVoiD-DF and much of the AV-deepfake literature is here; forensic framing, per-modality attribution, generalization protocol all match. **Primary target.** | Very competitive; expects the cross-dataset/LOGO story to hold up. Page limit ~14 two-column pages (≈ the 50–60 p. single-column draft restructured). |
| **Information Fusion** | Elsevier | Multimodal fusion is the journal's core identity; the disentangled-fusion + missing-modality design is a direct fit. Very high impact. | Highest desk-rejection bar; emphasize the *fusion* contribution, not only forensics. |
| **Pattern Recognition** | Elsevier | Publishes deepfake-detection and contrastive-representation papers regularly; QACP as a representation-learning contribution fits. | Long review cycles. |
| **IEEE Trans. on Multimedia (TMM)** | IEEE | Audio-visual analysis venue; localization + deployed system valued. | Fit slightly weaker than TIFS for the forensic claim. |

## 3. Tier 2 — solid Q1/Q2 applied venues (good acceptance odds)

| Journal | Publisher | Notes |
|---|---|---|
| **Expert Systems with Applications** | Elsevier | Values complete, deployed systems; the HF Space + UI + economic analysis is an asset here. |
| **Engineering Applications of Artificial Intelligence** | Elsevier | Applied AI framing; fast-growing deepfake output. |
| **Knowledge-Based Systems** | Elsevier | Representation-learning + application balance. |
| **Applied Soft Computing** | Elsevier | Published AVFakeNet (our baseline); reviewers will know the area. |
| **Computer Vision and Image Understanding** | Elsevier | Published BA-TFD+ / LAV-DF work; localization results directly comparable. |
| **Neurocomputing** | Elsevier | Broad ML venue, reasonable turnaround. |
| **ACM Trans. on Multimedia Computing, Communications and Applications (TOMM)** | ACM | Multimedia forensics track record. |

## 4. Tier 3 — fast / fallback (still Scopus)

| Journal | Notes |
|---|---|
| **IEEE Access** | Fast (~4–8 weeks), open access APC; respectable fallback, weaker signal for a Q1-targeting thesis. |
| **Forensic Science International: Digital Investigation** | Perfect domain fit, smaller audience; good if the forensic-workflow angle is emphasized. |
| **Journal of Visual Communication and Image Representation** | Reasonable fit, lower bar. |
| **EURASIP Journal on Information Security** | Open access, forensics scope. |

## 5. Recommended strategy

1. **TIFS first.** Restructure the thesis chapters into the standard IEEE
   two-column format (the chapters were deliberately written journal-style —
   see HANDOFF.md §8 item 8): drop OBE-specific sections (planning,
   economic decision, author contributions), expand related work with the
   Ch.1 material, keep cross-dataset/LOGO tables as the centerpiece.
2. If TIFS rejects with reviews, revise and go to **Information Fusion**
   (reframe around disentangled multimodal fusion) or **Pattern
   Recognition** (reframe around QACP as contrastive representation
   learning).
3. If a faster decision is needed for graduation timelines, submit in
   parallel-safe sequence to **ESWA or EAAI** (never simultaneously —
   Elsevier checks).
4. Before any submission: replace every blue `\dummy{}` value with measured
   results, regenerate figures from real `results/*.json`, add 2–4
   citations from the target journal's 2025–2026 issues, and run the
   plagiarism check required by AIUB.

## 6. Journal-finder tools (for double-checking scope)

- Elsevier JournalFinder — paste the abstract, filter "Scopus indexed."
- IEEE Publication Recommender.
- Scopus Sources list (scopus.com/sources) — confirm active indexing and
  CiteScore quartile for the exact year of submission.
