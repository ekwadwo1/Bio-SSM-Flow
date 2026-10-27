# Bio-SSM-Flow — External Validation Analysis Plan

| Field | Value |
|---|---|
| Version | v0.3 (DRAFT — not yet frozen) |
| Drafted | 2026-10-02 (v0.1: 2026-09-29; v0.2: 2026-10-01) |
| Cohorts | UCSF-ALPTDG (`UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0`), MU-Glioma-Post (TCIA version 1, updated 2025-03-21) |
| Frozen on | _to be filled at freeze_ |
| Freeze commit hash | _to be filled at freeze_ |

---

## Changes from v0.2

1. **Eligibility restricted to diffuse glioma** (Section 3). In MU-Glioma-Post this excludes the 3 patients with pilocytic astrocytoma. Decided on histology alone, before any forecast was computed.
2. **Pre-specified sensitivity analysis on registration quality** (Section 7): primary endpoints repeated after excluding the lowest decile of per-patient registration Dice in each cohort.
3. **Registration method specified as cohort-template registration** (Section 4): each cohort's mean T1 template is registered to a LUMIERE mean T1 template on the MNI152 grid; no external SRI24 template file is used, because the cohorts' headers follow the BraTS/CaPTk convention.
4. **Left–right orientation check added** to the registration QC (Section 4).
5. **MU-Glioma-Post ingestion completed** (pipeline QC only); its Appendix A items are filled in.
6. **UCSF-ALPTDG identity verified** from the archive contents; ingestion items pending.

---

## Rule 0 — Freeze before inference

No model or baseline inference on external data is performed until **this plan, the evaluation code, `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, and `runs/FROZEN_T_full_fold.json` are committed** and the commit hash is recorded above.

- **Allowed before freeze (pipeline QC):** file inventory, header/affine checks, label-integer inspection, label-semantics verification, registration quality control, VQGAN encoding and round-trip fidelity.
- **Not allowed before freeze:** computing any forecast, baseline prediction, or any metric derived from one; inspecting clinical outcome variables in relation to model output.

---

## 1. Hypotheses (primary, tested separately in each cohort)

- **H1a — accuracy vs. AR(1).** The flow model's ensemble-mean forecast has lower tumor-ROI latent MSE than the voxelwise AR(1) forecast.
- **H1b — distributional skill vs. AR(1).** The flow model's calibrated tumor-ROI CRPS is lower than the AR(1) forecaster's.
- **H2 — calibration transfer.** Applying the LUMIERE-fitted temperature *unchanged* lowers calibration error (brain region, against the finite-R ideal) relative to no calibration (T = 1).

---

## 2. Frozen components

| Component | Specification |
|---|---|
| Forecasting models | Flow V2, fold models 0–4, EMA weights |
| Model SHA-256 (prefix) | fold0 `bda53a14cc36` · fold1 `92af8aa55d8b` · fold2 `c9c9e33b8a95` · fold3 `1524d49430c8` · fold4 `ce239c4ab756` |
| Comparison baselines | Per fold k: voxelwise AR(1) and damped persistence fitted on fold k's LUMIERE training pairs (`runs/FROZEN_BASELINES.pt`) |
| Latent autoencoder | Frozen Paper-1 VQGAN (`vqgan_r4/best_vqgan.pt`) |
| Segmentation head | `runs/seghead_img_fold0/imgseg_best.pt` (hash in Appendix A) |
| Flow temperature T_k | Full-fold LUMIERE fit (brain region, finite-R ideal, grid step 0.05); values in Appendix A |
| Baseline temperature | AR(1) and damped persistence: T = 1 (their LUMIERE cross-fitted temperatures were 0.90–1.00) |
| Sampling | R = 32 ensemble members, 20 midpoint ODE steps, per-pair seed as defined in `eval_unified.py`; baselines: R = 32 Gaussian samples |
| Point forecasts | Latent metrics: ensemble mean (flow), analytic predictive mean (baselines). Image metrics: image-space mean of 8 decoded (uncalibrated) members |

---

## 3. Cohorts, eligibility, and pairs

| | UCSF-ALPTDG | MU-Glioma-Post |
|---|---|---|
| Patients in release | 298 | 203 |
| Pairs | One per patient (timepoint 1 → 2) | All consecutive post-treatment timepoint pairs |
| Source space | SRI24, 1 mm (BraTS/CaPTk header convention) | SRI24, 1 mm (BraTS/CaPTk header convention) |

**Eligibility (both cohorts):**

- **Diffuse glioma only** (WHO grade 2–4 diffuse astrocytic or oligodendroglial tumors, including glioblastoma). Non-diffuse histologies are excluded. In MU-Glioma-Post, the `Primary Diagnosis` categories GBM, Astrocytoma, Diffuse glioma, Oligodendro-glioma, and Glioma w/ GBM features are eligible; Pilocytic astrocytoma (3 patients) is excluded. UCSF-ALPTDG consists of diffuse gliomas by design.

**Pair exclusions (both cohorts; all counted and reported):**

- any of the four contrasts (T1, T1c, T2, FLAIR) or the timepoint segmentation missing;
- interval unknown or Δt < 7 days (same rule as LUMIERE);
- baseline acquired before the first surgery or procedure;
- failed registration QC (Section 4).

**Interval Δt.** The true interval is used where available; otherwise the LUMIERE median (84 days). This is justified by the pre-established internal finding that Δt has no measurable effect on the forecast (true vs. shuffled vs. constant Δt; whole volume and tumor ROI; all five folds; also with a direct, ungated Δt pathway). Where available, Δt is used for subgroup analysis.

**Note on LUMIERE.** 74 of the 469 internal pairs start at the pre-operative scan (week 0). Both external cohorts are post-treatment only; internal results are additionally reported for post-operative pairs only.

---

## 4. Preprocessing and registration (identical to training)

1. **Target grid.** FSL MNI152 1 mm: shape 182×218×182, affine `[[-1,0,0,90],[0,1,0,-126],[0,0,1,-72]]`, orientation LAS — the exact LUMIERE grid (verified 2026-09-30).
2. **Templates.** Reference: mean skull-stripped T1 of the 91 LUMIERE patients (first complete timepoint each), on the target grid. Cohort template: mean pre-contrast T1 of the cohort's patients (first complete timepoint each), on the cohort's native grid. Each image is scaled by its brain mean before averaging.
3. **Spatial mapping.** A single affine transform (12 DOF, ANTs, seed 2026) from cohort template to reference, applied identically to every volume of every timepoint (images: linear; labels: nearest neighbour). Patients failing the per-patient gate receive a per-patient affine (first-timepoint pre-contrast T1 → reference), logged; the better of the two is kept.
4. **Hard check.** Every output's shape and affine must equal the LUMIERE reference exactly, and the in-memory array must equal the written file; otherwise the pipeline stops.
5. **Registration QC.** Per-patient brain-mask Dice against the reference mask ≥ 0.90; visual overlay check on 20 random patients and the 10 lowest-Dice patients per cohort; left–right orientation check (cohort template registered as-is and mirrored; the as-is orientation must not correlate worse with the reference).
6. **Intensity and resampling.** Channel order [T1pre, T1post, FLAIR, T2]; the unchanged `transforms.py` pipeline (trilinear resize to 128³, per-scan per-channel z-score over nonzero voxels — the normalisation the frozen models were trained with); frozen VQGAN encoder; latents cached.
7. **Labels.** Each cohort's integers mapped to {0 background, 1 NETC, 2 SNFH, 3 ET}; resection cavity (RC) folded into background. Semantics verified on 10 timepoints per cohort (ET brightest on T1c when present).
8. **VQGAN round-trip fidelity.** Whole-volume and brain-voxel PSNR reported per cohort alongside LUMIERE. Reported only; not an exclusion criterion.

---

## 5. Model application

- **Patient-to-model assignment.** Each external patient is assigned to exactly one fold model k by a seeded, balanced random permutation (seed 2026, stratified by cohort), after eligibility filtering. The assignment file is saved and committed before inference.
- **Matched baselines.** Each patient's forecasts are compared with the AR(1) and damped-persistence baselines of the same fold k.
- **Temperature.** Flow predictions use T_k; baselines use T = 1.
- **Other baselines.** Copy (both cohorts); capped Linear (MU-Glioma-Post pairs with a preceding timepoint; extrapolation ratio capped at 1).
- **DDPM transport.** Not evaluated externally (proposed: the transport comparison is an internal efficiency result; confirm in Appendix A).

---

## 6. Endpoint definitions

**Tumor ROI.** Ground-truth ET ∪ NETC at baseline or follow-up, dilated by 3 voxels, pooled to latent tokens with > 10% occupancy. Sensitivity ROI: ET ∪ NETC ∪ SNFH.

**Calibration error.** Mean over nominal levels {0.1, 0.2, …, 0.9, 0.95} of |empirical coverage − finite-R ideal coverage|, over brain tokens.

**Primary endpoints**

- H1a: tumor-ROI latent MSE, flow vs. AR(1).
- H1b: calibrated tumor-ROI CRPS, flow vs. AR(1).
- H2: brain calibration error, T_k vs. T = 1.

**Secondary endpoints**

- Brain and whole-volume latent MSE vs. AR(1) (internally, AR(1) was equal or slightly better outside the tumor; reported as such).
- Tumor-ROI cosine (direction) vs. AR(1).
- All primary accuracy endpoints repeated against damped persistence and Copy.
- Raw (uncalibrated) CRPS vs. AR(1).
- Error–uncertainty Spearman, token level (whole, brain, ROI).
- Case-level uncertainty: Spearman between each patient's mean ROI uncertainty and ROI error.
- **UCSF-ALPTDG only:** voxel AUC of predicted uncertainty for the expert change annotations (union of the FLAIR and T1ce subtraction segmentations; a token counts as changed if ≥ 10% of its voxels lie in the annotation; AUC over brain tokens per patient, then averaged).

**Descriptive only (no hypothesis tests)**

- PSNR / SSIM on brain voxels (image-space mean of 8 decoded members vs. decoded follow-up), with Copy.
- Forecast-segmentation Dice vs. pipeline Copy (segmentation of the decoded baseline) and ceiling (segmentation of the decoded true follow-up).

---

## 7. Statistics

- **Unit of analysis: patient.** MU-Glioma-Post pairs are averaged per patient before testing.
- **H1a, H1b:** two-sided Wilcoxon signed-rank tests, flow vs. AR(1).
- **H2:** patient-bootstrap test of the difference in calibration error (5,000 resamples, seed 0).
- **Multiplicity:** Holm correction across the six primary tests (H1a, H1b, H2 × two cohorts); separately, Holm within the secondary family.
- **Effect sizes:** 95% patient-bootstrap confidence intervals of the relative improvement; rank-biserial correlation for Wilcoxon tests.
- **Sensitivity analysis (pre-specified):** the three primary endpoints are repeated in each cohort after excluding patients in the lowest decile of per-patient registration Dice. Reported alongside the primary results with confidence intervals; not part of the Holm family and not used to support primary claims.
- **Every pre-specified outcome is reported, regardless of result.**

---

## 8. Pre-specified exploratory subgroups

- Histology: GBM vs. other diffuse glioma.
- Δt inside vs. outside the LUMIERE 5th–95th percentile range (7–175 days).
- Time since surgery: median split (UCSF-ALPTDG).
- UCSF-ALPTDG expert change category (increase / decrease / unchanged / mixed).

Subgroup results are reported descriptively with confidence intervals; they are not used to support primary claims.

---

## 9. Deviations

- Every deviation from this plan is recorded below with date and reason.
- Settings are never changed and rerun to improve results.
- If a bug is found, it is fixed, the **entire affected cohort is rerun**, and both runs are reported.

### Deviations log

| Date | Section | Deviation | Reason |
|---|---|---|---|
| | | | |

---

## 10. Reporting

All reported numbers, tables, and figure data are generated by scripts from saved JSON outputs. No number is entered by hand.

---

## Appendix A — Frozen values and cohort details

### Frozen models and calibration

| Item | Value |
|---|---|
| T_0 … T_4 (full-fold LUMIERE fit) | 1.35, 1.35, 1.20, 1.40, 1.40 (`runs/FROZEN_T_full_fold.json`) |
| `FROZEN_BASELINES.pt` SHA-256 | `56ec59135e54d190772e8880bfca09f884361a0668fb3e4999574f7195c8c347` |
| Segmentation-head SHA-256 | _from `runs/eval_unified/flow/meta.json` → `seg_head.sha256`_ |
| MNI152 reference | LUMIERE mean skull-stripped T1, 91 patients (`external/mu/lumiere_ref_T1.nii.gz`) |
| DDPM external evaluation included? | No (proposed) — confirm |
| External-mode evaluation code commit | _to complete_ |
| Patient-to-model assignment file (path + SHA-256) | _to complete_ |

### MU-Glioma-Post (ingestion completed 2026-10-02; pipeline QC only)

| Item | Value |
|---|---|
| Contrast mapping | T1pre = `brain_t1n`, T1post = `brain_t1c`, FLAIR = `brain_t2f`, T2 = `brain_t2w` |
| Label mapping (raw → unified) | `brats2024` crosswalk: 1 NETC, 2 SNFH, 3 ET, 4 RC → 0 background |
| Label semantics | ET brightest on T1c in 10/10 checked timepoints |
| Interval field | `Number of Days from Diagnosis to kth MRI (Timepoint_k)`; Δt = day(b) − day(a) |
| Surgery field | `Number of days from Diagnosis to First surgery or procedure` |
| Timepoints | 596 with images; 594 complete (2 without segmentation) |
| Registration | Cohort template from 203 patients; template Dice 0.984; per-patient Dice median 0.929, minimum 0.901, none < 0.90; per-patient fallback used for 31 patients |
| Geometry hard check | PASS |
| Left–right orientation check | as-is 0.9565 > mirrored 0.9475 (correct orientation) |
| Visual QC | 20 random + 10 lowest-Dice patients: no misalignment; edge mismatches from brain size/shape only |
| VQGAN round-trip PSNR | whole 32.13 dB, brain 26.97 dB (LUMIERE: 33.31 / 26.99) |
| Pairs | 391 consecutive; 377 meeting interval, surgery, and registration criteria (151 patients), before the histology rule; final eligible count computed by the evaluation script |
| Ingestion outputs | `external/mu/` (`mu_pairs.json`, `ingest_summary.json`, overlays) and `latent_cache_mu/` |

### UCSF-ALPTDG (identity verified 2026-10-02; ingestion pending)

| Item | Value |
|---|---|
| Archive | `UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip` (SHA-256 in `ucsf-alptdg/DOWNLOAD_SHA256.txt`) |
| Content check | 298 patient folders; time1/time2 for each; 596 each of T1, T1ce, T2, FLAIR, segmentation; 298 FLAIR and 298 T1ce subtraction images with change segmentations; Table S1 |
| Contrast mapping | T1pre = `_timeN_t1`, T1post = `_timeN_t1ce`, FLAIR = `_timeN_flair`, T2 = `_timeN_t2` |
| Label mapping (raw → unified) | _to complete after inspection_ |
| Change-annotation coding | _to complete after inspection_ |
| Interval field (Table S1 column) | _to complete after inspection_ |
| Registration QC, orientation check, VQGAN fidelity, pair counts | _to complete after ingestion_ |

## Appendix B — Freeze checklist

- [ ] Appendix A complete (both cohorts)
- [ ] `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, `runs/FROZEN_T_full_fold.json` committed
- [ ] Evaluation code (external mode) committed
- [ ] Patient-to-model assignment file committed
- [ ] Freeze commit hash recorded in the header table
- [ ] Only then: run inference
