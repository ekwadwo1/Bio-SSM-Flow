# Bio-SSM-Flow — External Validation Analysis Plan

| Field | Value |
|---|---|
| Version | v0.4 (DRAFT — not yet frozen) |
| Drafted | 2026-10-02 (v0.1: 2026-09-29; v0.2: 2026-10-01; v0.3: 2026-10-02) |
| Cohorts | UCSF-ALPTDG (`UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0`), MU-Glioma-Post (TCIA version 1, updated 2025-03-21) |
| Frozen on | _to be filled at freeze_ |
| Freeze commit hash | _to be filled at freeze_ |

---

## Changes from v0.3

1. **Eligibility rule made explicit and identical for both cohorts:** tumors with a confirmed circumscribed (non-diffuse) histology are excluded (Section 3). This removes 3 MU-Glioma-Post patients (pilocytic astrocytoma) and 4 UCSF-ALPTDG patients (1 pilocytic astrocytoma, 3 pleomorphic xanthoastrocytoma).
2. **UCSF-ALPTDG baseline-before-surgery exclusion applied:** 2 patients (100088, 100294).
3. **UCSF-ALPTDG time-since-surgery values that are not valid intervals** (spreadsheet date serials or implausible values; 8 patients) are treated as missing for the time-since-surgery subgroup only.
4. **Two additional pre-specified UCSF-ALPTDG sensitivity analyses:** excluding pairs with an intervening surgery, and excluding the patients also present in UCSF-PDGM (Section 7).
5. **UCSF-ALPTDG ingestion completed** (pipeline QC only); Appendix A filled.

---

## Rule 0 — Freeze before inference

No model or baseline inference on external data is performed until **this plan, the evaluation code, `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, and `runs/FROZEN_T_full_fold.json` are committed** and the commit hash is recorded above.

- **Allowed before freeze (pipeline QC):** file inventory, header/affine checks, label-integer inspection, label-semantics verification, registration quality control, VQGAN encoding and round-trip fidelity, eligibility and timing fields.
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
| Latent autoencoder | Frozen Paper-1 VQGAN (`vqgan_r4/best_vqgan.pt`), trained on BraTS 2021 |
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
| Pairs | One per patient (time 1 → time 2) | All consecutive post-treatment timepoint pairs |
| Source space | SRI24, 1 mm | SRI24, 1 mm |
| Eligible (after all rules) | 292 patients, 292 pairs | computed by the evaluation script (377 pairs from 151 patients before the histology rule) |

**Eligibility (identical in both cohorts).** Diffuse gliomas are eligible. Tumors with a **confirmed circumscribed (non-diffuse) histology** under the WHO 2021 classification are excluded.

- MU-Glioma-Post: `Primary Diagnosis` = Pilocytic astrocytoma (3 patients) excluded. GBM, Astrocytoma, Diffuse glioma, Oligodendro-glioma, and Glioma w/ GBM features are eligible.
- UCSF-ALPTDG: patients 100011 (pilocytic astrocytoma) and 100027, 100153, 100190 (pleomorphic xanthoastrocytoma) excluded. Presumed low-grade glioma without histological confirmation (100007, 100033), diffuse midline glioma (100183), and all other diagnoses (glioblastoma, gliosarcoma, glioma NOS, oligodendroglioma, IDH-mutant and IDH-wild-type astrocytoma, IDH-mutant NOS) are eligible.

**Pair exclusions (both cohorts; all counted and reported):**

- any of the four contrasts (T1, T1c, T2, FLAIR) or the timepoint segmentation missing;
- interval unknown or Δt < 7 days (same rule as LUMIERE);
- baseline acquired before the first surgery or procedure (UCSF-ALPTDG: 100088, 100294 — negative `Days from 1st surgery/DX to 1st scan`);
- failed registration QC (Section 4).

**Interval Δt.** The true interval is used where available; otherwise the LUMIERE median (84 days). Justified by the pre-established internal finding that Δt has no measurable effect on the forecast. Where available, Δt is used for subgroup analysis.

**Note on LUMIERE.** 74 of the 469 internal pairs start at the pre-operative scan (week 0). Both external cohorts are post-treatment only; internal results are additionally reported for post-operative pairs only.

---

## 4. Preprocessing and registration (identical to training)

1. **Target grid.** FSL MNI152 1 mm: shape 182×218×182, affine `[[-1,0,0,90],[0,1,0,-126],[0,0,1,-72]]`, orientation LAS — the exact LUMIERE grid.
2. **Templates.** Reference: mean skull-stripped T1 of the 91 LUMIERE patients (first complete timepoint each). Cohort template: mean pre-contrast T1 of the cohort's patients (first timepoint each), on the cohort's native grid. Each image is scaled by its brain mean before averaging. Each cohort has its own template and transform (the cohorts use different header origin conventions).
3. **Spatial mapping.** A single affine transform (12 DOF, ANTs, seed 2026) from cohort template to reference, applied identically to every volume of every timepoint (images: linear; labels and change annotations: nearest neighbour). Patients below the per-patient gate receive a per-patient affine (first-timepoint pre-contrast T1 → reference), logged; the better of the two is kept.
4. **Hard check.** Every output's shape and affine must equal the LUMIERE reference exactly, and the in-memory array must equal the written file; otherwise the pipeline stops.
5. **Registration QC.** Per-patient brain-mask Dice against the reference mask ≥ 0.90; visual overlay check on 20 random patients and the 10 lowest-Dice patients per cohort; left–right orientation check.
6. **Intensity and resampling.** Channel order [T1pre, T1post, FLAIR, T2]; the unchanged `transforms.py` pipeline (trilinear resize to 128³, per-scan per-channel z-score over nonzero voxels); frozen VQGAN encoder; latents cached.
7. **Labels.** Mapped to {0 background, 1 NETC, 2 SNFH, 3 ET}; resection cavity folded into background. Semantics verified on 10 timepoints per cohort.
8. **VQGAN round-trip fidelity.** Whole-volume and brain-voxel PSNR reported per cohort alongside LUMIERE. Reported only; not an exclusion criterion. Differences in fidelity are taken into account when interpreting between-cohort differences in forecasting performance.

---

## 5. Model application

- **Patient-to-model assignment.** Each eligible external patient is assigned to exactly one fold model k by a seeded, balanced random permutation (seed 2026, stratified by cohort). The assignment file is saved and committed before inference.
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
- **UCSF-ALPTDG only:** voxel AUC of predicted uncertainty for the expert change annotations (any nonzero value in the FLAIR or T1ce change annotation; a token counts as changed if ≥ 10% of its voxels are annotated; AUC over brain tokens per patient, then averaged).

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
- **Pre-specified sensitivity analyses** (primary endpoints repeated; reported with confidence intervals; not part of the Holm family; not used to support primary claims):
  1. **Registration quality (both cohorts):** excluding patients in the lowest decile of per-patient registration Dice.
  2. **Intervening surgery (UCSF-ALPTDG):** excluding pairs with a surgery between the two scans (`Days from 1st scan to 2nd SX` between 0 and Δt; 6 pairs before other exclusions).
  3. **UCSF-PDGM overlap (UCSF-ALPTDG):** excluding the 60 patients also present in UCSF-PDGM, whose pre-operative scans may have been part of the BraTS 2021 data used to train the frozen VQGAN.
- **Every pre-specified outcome is reported, regardless of result.**

---

## 8. Pre-specified exploratory subgroups

- Histology: glioblastoma (including gliosarcoma) vs. other diffuse glioma.
- Δt inside vs. outside the LUMIERE 5th–95th percentile range (7–175 days).
- Time since surgery: median split (UCSF-ALPTDG; the 8 patients with invalid values are excluded from this subgroup only).
- UCSF-ALPTDG expert change category (Table S1 `SNFH_CHANGE_OVERALL`, `ET_CHANGE_OVERALL`).

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

### MU-Glioma-Post (ingestion completed 2026-10-02)

| Item | Value |
|---|---|
| Contrast mapping | T1pre = `brain_t1n`, T1post = `brain_t1c`, FLAIR = `brain_t2f`, T2 = `brain_t2w` |
| Label mapping (raw → unified) | `brats2024` crosswalk: 1 NETC, 2 SNFH, 3 ET, 4 RC → 0 |
| Label semantics | ET brightest on T1c in 10/10 checked timepoints |
| Interval field | `Number of Days from Diagnosis to kth MRI (Timepoint_k)`; Δt = day(b) − day(a) |
| Surgery field | `Number of days from Diagnosis to First surgery or procedure` |
| Timepoints | 596 with images; 594 complete |
| Registration | Template from 203 patients; template Dice 0.984; per-patient Dice median 0.929, minimum 0.901, none < 0.90; per-patient affine used for 31 patients |
| Geometry hard check | PASS |
| Orientation check | as-is 0.9565 > mirrored 0.9475 |
| Visual QC | No misalignment; edge mismatches from brain size/shape only |
| VQGAN round-trip PSNR | whole 32.13 dB, brain 26.97 dB (LUMIERE: 33.31 / 26.99) |
| Pairs | 391 consecutive; 377 meeting interval, surgery, and registration criteria (151 patients) before the histology rule |
| Δt (usable pairs) | median 77 days, 5–95%: 18–203 |

### UCSF-ALPTDG (ingestion completed 2026-10-02)

| Item | Value |
|---|---|
| Archive | `UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip` (SHA-256 in `ucsf-alptdg/DOWNLOAD_SHA256.txt`) |
| Contrast mapping | T1pre = `_timeN_t1`, T1post = `_timeN_t1ce`, FLAIR = `_timeN_flair`, T2 = `_timeN_t2` |
| Label mapping (raw → unified) | `brats2024` crosswalk: 1 NCR/NETC, 2 SNFH, 3 ET, 4 RC → 0 (Table S1: `NCR_Volume_Label1` … `RC_Volume_Label4`) |
| Label semantics | ET brightest on T1c in 10/10 checked timepoints |
| Change annotations | `flair_subtraction_seg` (SNFH) and `t1ce_subtraction_seg` (ET); 1 = increase, 2 = decrease — voxel counts match Table S1 increase/decrease volumes exactly (10 and 5 patients checked) |
| Interval field | Table S1 `Clinical Info`: `Days from 1st scan to 2nd scan` |
| Surgery field | `Days from 1st surgery/DX to 1st scan` (positive = scan after surgery; median 765 days) |
| Geometry | 240×240×155, 1 mm; time 1 and time 2 on an identical grid (20/20 checked) |
| Registration | Template from 298 patients; template Dice 0.984; per-patient Dice median 0.920, minimum 0.900, none < 0.90; per-patient affine used for 113 patients |
| Geometry hard check | PASS |
| Orientation check | as-is 0.9719 > mirrored 0.9687 |
| Visual QC | No misalignment; in some cohort-transform cases the brain sits at a slightly different superior–inferior level or tilt than the reference (frontal gap; Dice 0.90–0.92) — covered by sensitivity analysis 1 |
| VQGAN round-trip PSNR | whole 30.62 dB, brain 25.19 dB (about 1.8 dB below LUMIERE and MU-Glioma-Post) |
| Δt | median 65 days, 5–95%: 30–204; none missing; none < 7 days |
| Exclusions | 4 non-diffuse histology (100011, 100027, 100153, 100190); 2 baseline before surgery (100088, 100294) |
| Eligible | 292 patients, 292 pairs |
| Intervening surgery | 6 pairs (before other exclusions) |
| UCSF-PDGM overlap | 60 patients |
| Invalid time-since-surgery values | 100016, 100027, 100029, 100035, 100041, 100056, 100058, 100064 |

## Appendix B — Freeze checklist

- [ ] Appendix A complete (segmentation-head hash; DDPM decision; assignment file; code commit)
- [ ] `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, `runs/FROZEN_T_full_fold.json` committed
- [ ] Evaluation code (external mode) committed and smoke-tested on LUMIERE data only
- [ ] Patient-to-model assignment file committed
- [ ] Freeze commit hash recorded in the header table
- [ ] Only then: run inference
