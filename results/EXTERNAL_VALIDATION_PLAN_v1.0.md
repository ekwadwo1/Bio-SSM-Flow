# Bio-SSM-Flow — External Validation Analysis Plan

| Field | Value |
|---|---|
| Version | **v1.0 — FROZEN** |
| Frozen on | 2026-10-05 |
| Freeze commit | recorded in `FREEZE_RECORD.txt` and git tag `ext-validation-freeze` (a document cannot contain the hash of the commit that contains it) |
| History | v0.1 2026-09-29 · v0.2 2026-10-01 · v0.3–v0.4 2026-10-02 · v1.0 2026-10-05 |
| Cohorts | UCSF-ALPTDG (`UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0`), MU-Glioma-Post (TCIA version 1, updated 2025-03-21) |

---

## Changes from v0.4

1. Appendix A completed: segmentation-head hash, DDPM decision, eligible counts for both cohorts, patient-to-model assignment and its hash, evaluation code.
2. MU-Glioma-Post eligibility clarified: the 3 pilocytic patients are identified from the clinical table; 2 of them had pairs (6 pairs), 1 had none.

No change to hypotheses, endpoints, comparators, statistics, sensitivity analyses, or subgroups.

---

## Rule 0 — Freeze before inference

No model or baseline inference on external data is performed before this plan, the evaluation code (`eval_external.py` and the modules it imports), `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, `runs/FROZEN_T_full_fold.json`, and `runs/external_eval/assignment.json` are committed and the freeze commit hash is recorded. The evaluation script refuses to run on external cohorts without that hash and verifies that the eligible patients match the committed assignment.

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
| Latent autoencoder | Frozen Paper-1 VQGAN (`vqgan_r4/best_vqgan.pt`), trained on BraTS 2021 |
| Segmentation head | `runs/seghead_img_fold0/imgseg_best.pt` |
| Flow temperature T_k | Full-fold LUMIERE fit (brain region, finite-R ideal, grid step 0.05) |
| Baseline temperature | AR(1) and damped persistence: T = 1 (their LUMIERE cross-fitted temperatures were 0.90–1.00) |
| Sampling | R = 32 ensemble members, 20 midpoint ODE steps, per-pair seed derived from cohort, patient, and timepoints; baselines: R = 32 Gaussian samples |
| Point forecasts | Latent metrics: ensemble mean (flow), analytic predictive mean (baselines). Image metrics: image-space mean of 8 decoded (uncalibrated) members |

---

## 3. Cohorts, eligibility, and pairs

| | UCSF-ALPTDG | MU-Glioma-Post |
|---|---|---|
| Patients in release | 298 | 203 |
| Pairs | One per patient (time 1 → time 2) | All consecutive post-treatment timepoint pairs |
| Source space | SRI24, 1 mm | SRI24, 1 mm |
| **Eligible** | **292 patients, 292 pairs** | **149 patients, 371 pairs** |

**Eligibility (identical in both cohorts).** Diffuse gliomas are eligible. Tumors with a **confirmed circumscribed (non-diffuse) histology** under the WHO 2021 classification are excluded.

- MU-Glioma-Post: `Primary Diagnosis` = Pilocytic astrocytoma — MU-0189, MU-0206, MU-0208 excluded (two contributed pairs, 6 pairs in total; one contributed none).
- UCSF-ALPTDG: 100011 (pilocytic astrocytoma) and 100027, 100153, 100190 (pleomorphic xanthoastrocytoma) excluded. Presumed low-grade glioma without histological confirmation (100007, 100033), diffuse midline glioma (100183), and all other diagnoses are eligible.

**Pair exclusions (both cohorts):**

- any of the four contrasts (T1, T1c, T2, FLAIR) or the timepoint segmentation missing;
- interval unknown or Δt < 7 days;
- baseline acquired before the first surgery or procedure (UCSF-ALPTDG: 100088, 100294);
- failed registration QC (Section 4).

**Interval Δt.** The true interval is used where available; otherwise the LUMIERE median (84 days), justified by the pre-established internal finding that Δt has no measurable effect on the forecast. Where available, Δt is used for subgroup analysis.

**Note on LUMIERE.** 74 of the 469 internal pairs start at the pre-operative scan (week 0); internal results are additionally reported for post-operative pairs only.

---

## 4. Preprocessing and registration (identical to training)

1. **Target grid.** FSL MNI152 1 mm: shape 182×218×182, affine `[[-1,0,0,90],[0,1,0,-126],[0,0,1,-72]]`, orientation LAS — the exact LUMIERE grid.
2. **Templates.** Reference: mean skull-stripped T1 of the 91 LUMIERE patients. Cohort template: mean pre-contrast T1 of the cohort's patients (first timepoint each). Each cohort has its own template and transform.
3. **Spatial mapping.** A single affine transform (12 DOF, ANTs, seed 2026) from cohort template to reference, applied identically to every volume of every timepoint (images: linear; labels and change annotations: nearest neighbour). Patients below the per-patient gate receive a per-patient affine; the better of the two is kept.
4. **Hard check.** Every output's shape and affine must equal the LUMIERE reference exactly; otherwise the pipeline stops.
5. **Registration QC.** Per-patient brain-mask Dice ≥ 0.90; visual overlay check on 20 random and the 10 lowest-Dice patients per cohort; left–right orientation check.
6. **Intensity and resampling.** Channel order [T1pre, T1post, FLAIR, T2]; the unchanged `transforms.py` pipeline (trilinear resize to 128³, per-scan per-channel z-score over nonzero voxels); frozen VQGAN encoder; latents cached.
7. **Labels.** Mapped to {0 background, 1 NETC, 2 SNFH, 3 ET}; resection cavity folded into background. Semantics verified on 10 timepoints per cohort.
8. **VQGAN round-trip fidelity.** Reported per cohort alongside LUMIERE; not an exclusion criterion. Differences in fidelity are taken into account when interpreting between-cohort differences in forecasting performance.

---

## 5. Model application

- **Patient-to-model assignment.** Each eligible patient is assigned to one fold model by a seeded, balanced random permutation (seed 2026, per cohort). Fixed in `runs/external_eval/assignment.json` (Appendix A).
- **Matched baselines.** Each patient's forecasts are compared with the AR(1) and damped-persistence baselines of the same fold.
- **Temperature.** Flow predictions use T_k of the assigned fold; baselines use T = 1.
- **Other baselines.** Copy (both cohorts); capped Linear (MU-Glioma-Post pairs with a preceding timepoint; extrapolation ratio capped at 1).
- **DDPM transport.** Not evaluated externally.

---

## 6. Endpoint definitions

**Tumor ROI.** Ground-truth ET ∪ NETC at baseline or follow-up, dilated by 3 voxels, pooled to latent tokens with > 10% occupancy. Sensitivity ROI: ET ∪ NETC ∪ SNFH.

**Calibration error.** Mean over nominal levels {0.1, 0.2, …, 0.9, 0.95} of |empirical coverage − finite-R ideal coverage|, over brain tokens; coverage curves are averaged per patient, then over patients.

**Primary endpoints:** H1a tumor-ROI latent MSE (flow vs. AR(1)); H1b calibrated tumor-ROI CRPS (flow vs. AR(1)); H2 brain calibration error (T_k vs. T = 1).

**Secondary endpoints:** brain and whole-volume latent MSE vs. AR(1); tumor-ROI cosine vs. AR(1); raw tumor-ROI CRPS vs. AR(1); tumor-ROI MSE and calibrated CRPS vs. damped persistence; tumor-ROI MSE vs. Copy. Also reported: error–uncertainty Spearman (brain, ROI); case-level Spearman between per-patient mean ROI uncertainty and ROI error; **UCSF-ALPTDG only:** voxel AUC of predicted uncertainty for the expert change annotations (any nonzero FLAIR or T1ce change label; token counts as changed if ≥ 10% of its voxels are annotated; AUC over brain tokens per patient, then averaged).

**Descriptive only:** PSNR / SSIM on brain voxels (image-space mean of 8 decoded members vs. decoded follow-up), with Copy; forecast-segmentation Dice (ET, TC, WT) vs. pipeline Copy and ceiling.

---

## 7. Statistics

- **Unit of analysis: patient.** Pairs are averaged per patient before testing.
- **H1a, H1b:** two-sided Wilcoxon signed-rank tests; rank-biserial correlation; 95% patient-bootstrap confidence intervals of the relative improvement (5,000 resamples).
- **H2:** patient-bootstrap test of the difference in calibration error (5,000 resamples, seed 0), two-sided.
- **Multiplicity:** Holm correction across the six primary tests (H1a, H1b, H2 × two cohorts); separately, Holm across the secondary tests of both cohorts.
- **Pre-specified sensitivity analyses** (primary endpoints repeated; not part of the Holm family; not used to support primary claims):
  1. Registration quality (both cohorts): excluding the lowest decile (floor of 10%) of patients by registration Dice.
  2. Intervening surgery (UCSF-ALPTDG): excluding pairs with a surgery between the two scans.
  3. UCSF-PDGM overlap (UCSF-ALPTDG): excluding the 60 patients also present in UCSF-PDGM.
- **Every pre-specified outcome is reported, regardless of result.**

---

## 8. Pre-specified exploratory subgroups

Reported descriptively (relative improvement vs. AR(1) in tumor-ROI MSE and calibrated CRPS, with confidence intervals), not used to support primary claims:

- Histology: glioblastoma (MU: `GBM`; UCSF-ALPTDG: Glioblastoma, Gliosarcoma) vs. other diffuse glioma.
- Δt inside vs. outside the LUMIERE 5th–95th percentile range (7–175 days).
- Time since surgery: median split (UCSF-ALPTDG; 8 patients with invalid values excluded from this subgroup only).
- UCSF-ALPTDG expert SNFH change category (Table S1 `SNFH_CHANGE_OVERALL`).

---

## 9. Deviations

Every deviation after the freeze is recorded below with date and reason. Settings are never changed and rerun to improve results. If a bug is found, it is fixed, the entire affected cohort is rerun, and both runs are reported.

| Date | Section | Deviation | Reason |
|---|---|---|---|
| | | | |

---

## 10. Reporting

All reported numbers, tables, and figure data are generated by scripts from saved JSON outputs. No number is entered by hand.

---

## Appendix A — Frozen values and cohort details

### Frozen artifacts

| Item | Value |
|---|---|
| T_0 … T_4 (full-fold LUMIERE fit) | 1.35, 1.35, 1.20, 1.40, 1.40 (`runs/FROZEN_T_full_fold.json`) |
| `FROZEN_BASELINES.pt` SHA-256 | `56ec59135e54d190772e8880bfca09f884361a0668fb3e4999574f7195c8c347` |
| Segmentation-head SHA-256 | `15ba42d79e0d2711f686c2aca30272d68fd28e5533a6ea9a85f605e5b53cb5d0` |
| MNI152 reference | LUMIERE mean skull-stripped T1, 91 patients (`external/mu/lumiere_ref_T1.nii.gz`) |
| DDPM external evaluation | **No** — the transport comparison is an internal efficiency result; four of five internal DDPM models were not trained to convergence |
| Patient-to-model assignment | `runs/external_eval/assignment.json`, SHA-256 `6765630a4c0de6260fbcf0285f57c0b8fd8b25a1d5f4fc865dc608f639161f7c`; UCSF-ALPTDG 59/59/58/58/58, MU-Glioma-Post 30/30/30/30/29 patients per fold model |
| Evaluation code | `eval_external.py` (with `eval_unified.py`, `aggregate_unified.py`) at the freeze commit; smoke-tested on 10 LUMIERE pairs only (2026-10-02) |

### MU-Glioma-Post

| Item | Value |
|---|---|
| Contrast mapping | T1pre = `brain_t1n`, T1post = `brain_t1c`, FLAIR = `brain_t2f`, T2 = `brain_t2w` |
| Label mapping | `brats2024` crosswalk: 1 NETC, 2 SNFH, 3 ET, 4 RC → 0 |
| Label semantics | ET brightest on T1c in 10/10 checked timepoints |
| Interval field | `Number of Days from Diagnosis to kth MRI (Timepoint_k)`; Δt = day(b) − day(a) |
| Surgery field | `Number of days from Diagnosis to First surgery or procedure` |
| Registration | Template Dice 0.984; per-patient Dice median 0.929, minimum 0.901; per-patient affine for 31 patients; hard check PASS; orientation as-is 0.9565 > mirrored 0.9475 |
| VQGAN round-trip PSNR | whole 32.13 dB, brain 26.97 dB (LUMIERE: 33.31 / 26.99) |
| Pairs | 391 consecutive → 377 meeting interval, surgery, registration criteria → **371 eligible (149 patients)** after the histology rule |
| Δt | median 77 days, 5–95%: 18–203 (before the histology rule) |

### UCSF-ALPTDG

| Item | Value |
|---|---|
| Archive | `UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip` (SHA-256 in `ucsf-alptdg/DOWNLOAD_SHA256.txt`) |
| Contrast mapping | T1pre = `_timeN_t1`, T1post = `_timeN_t1ce`, FLAIR = `_timeN_flair`, T2 = `_timeN_t2` |
| Label mapping | `brats2024` crosswalk: 1 NCR/NETC, 2 SNFH, 3 ET, 4 RC → 0 |
| Label semantics | ET brightest on T1c in 10/10 checked timepoints |
| Change annotations | 1 = increase, 2 = decrease (both annotations; verified against Table S1 volumes) |
| Interval field | Table S1 `Clinical Info`: `Days from 1st scan to 2nd scan` |
| Surgery field | `Days from 1st surgery/DX to 1st scan` (positive = scan after surgery) |
| Registration | Template Dice 0.984; per-patient Dice median 0.920, minimum 0.900; per-patient affine for 113 patients; hard check PASS; orientation as-is 0.9719 > mirrored 0.9687 |
| VQGAN round-trip PSNR | whole 30.62 dB, brain 25.19 dB |
| Δt | median 65 days, 5–95%: 30–204 |
| Exclusions | 4 non-diffuse (100011, 100027, 100153, 100190); 2 baseline before surgery (100088, 100294) |
| **Eligible** | **292 patients, 292 pairs** |
| Intervening surgery | 6 pairs (before other exclusions) |
| UCSF-PDGM overlap | 60 patients |
| Invalid time-since-surgery values | 100016, 100027, 100029, 100035, 100041, 100056, 100058, 100064 |

## Appendix B — Freeze checklist

- [x] Appendix A complete
- [x] Frozen models, baselines, and temperatures recorded with hashes
- [x] Evaluation code smoke-tested on LUMIERE data only
- [x] Patient-to-model assignment created and hashed
- [ ] Plan, code, and assignment committed; commit tagged `ext-validation-freeze`; hash written to `FREEZE_RECORD.txt`
- [ ] Only then: run inference on UCSF-ALPTDG and MU-Glioma-Post
