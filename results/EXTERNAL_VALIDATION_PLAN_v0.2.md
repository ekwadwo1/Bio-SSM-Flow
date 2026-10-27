# Bio-SSM-Flow — External Validation Analysis Plan

| Field | Value |
|---|---|
| Version | v0.2 (DRAFT — not yet frozen) |
| Drafted | 2026-10-01 (v0.1: 2026-09-29) |
| Cohorts | UCSF-ALPTDG, MU-Glioma-Post (TCIA version 1, updated 2025-03-21) |
| Frozen on | _to be filled at freeze_ |
| Freeze commit hash | _to be filled at freeze_ |

---

## Changes from v0.1

1. **Primary accuracy endpoint changed from Copy to voxelwise AR(1).** Internal LUMIERE results showed that beating Copy mostly reflects regression to the mean; the AR(1) forecaster (and damped persistence) are the baselines that test genuine forecasting skill. Copy moves to the secondary endpoints.
2. **A second primary accuracy endpoint added:** calibrated tumor-ROI CRPS vs. AR(1).
3. **Frozen model confirmed:** the current flow V2 models. The pre-specified AR(1)-anchored pilot did not meet its criteria (2026-10-01; see `runs/eval_unified/ar1flow/pilot_criteria.json`).
4. **Comparison baselines frozen per fold** (`runs/FROZEN_BASELINES.pt`), each fitted on exactly the training pairs its fold's flow model saw.
5. **Temperatures T_k fixed** from the full-fold LUMIERE fit (Appendix A).
6. **Image and segmentation metrics are descriptive only**, against pipeline Copy; growth-direction accuracy removed.
7. **Preprocessing stated explicitly:** the original per-scan z-score normalisation (the one the frozen models were trained with), not the lesion-aware variant.

---

## Rule 0 — Freeze before inference

No model or baseline inference on external data is performed until **this plan, the evaluation code, `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, and `runs/FROZEN_T_full_fold.json` are committed** and the commit hash is recorded above.

- **Allowed before freeze (pipeline QC):** file inventory, header/affine checks, label-integer inspection, label-semantics verification, registration quality control, VQGAN encoding.
- **Not allowed before freeze:** computing any forecast, baseline prediction, or any metric derived from one.

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
| Point forecasts | Latent metrics: ensemble mean (flow), analytic predictive mean (baselines). Image metrics: image-space mean of 8 decoded samples |

---

## 3. Cohorts and pairs

| | UCSF-ALPTDG | MU-Glioma-Post |
|---|---|---|
| Pairs | One per patient (timepoint 1 → 2) | All consecutive post-treatment timepoint pairs |
| Source space | SRI24, 1 mm | SRI24, 1 mm |

**Exclusions (both cohorts; all counted and reported):**

- any of the four contrasts (T1, T1c, T2, FLAIR) missing;
- failed registration QC (Section 4);
- interval Δt < 7 days (same rule as LUMIERE).

**Interval Δt.** The true interval is used where available; otherwise the LUMIERE median (84 days). This is justified by the pre-established internal finding that Δt has no measurable effect on the forecast (true vs. shuffled vs. constant Δt; whole volume and tumor ROI; all five folds; also with a direct, ungated Δt pathway). Where available, Δt is used for subgroup analysis.

**Note on LUMIERE.** 74 of the 469 internal pairs start at the pre-operative scan (week 0). Both external cohorts are post-treatment only; internal results will additionally be reported for post-operative pairs only.

---

## 4. Preprocessing (identical to training)

1. **Target grid.** FSL MNI152 1 mm: shape 182×218×182, affine `[[-1,0,0,90],[0,1,0,-126],[0,0,1,-72]]`, orientation LAS — the exact LUMIERE grid (verified 2026-09-30).
2. **Spatial mapping.** A single SRI24 → MNI152 affine transform (12 DOF), computed once from the two templates, applied identically to both timepoints of every patient. Images: linear interpolation. Labels: nearest-neighbour. Per-patient affine refinement only for patients failing QC, logged.
3. **Hard check.** Every output volume's shape and affine must equal the LUMIERE reference exactly; otherwise the pipeline stops.
4. **Registration QC.** Brain-mask Dice against the MNI reference mask ≥ 0.90; visual overlay check on 20 randomly selected patients per cohort.
5. **Intensity and resampling.** Channel order [T1pre, T1post, FLAIR, T2]; the unchanged `transforms.py` pipeline (trilinear resize to 128³, per-scan per-channel z-score over nonzero voxels — the normalisation the frozen models were trained with); frozen VQGAN encoder; latents cached.
6. **Labels.** Each cohort's integers mapped to {0 background, 1 NETC, 2 SNFH, 3 ET}; resection cavity (RC) folded into background. Semantics verified on 10 cases per cohort (ET brightest on T1c when present).
7. **VQGAN round-trip fidelity.** Whole-volume and brain-voxel PSNR/SSIM reported per cohort alongside LUMIERE. Reported only; not an exclusion criterion.

---

## 5. Model application

- **Patient-to-model assignment.** Each external patient is assigned to exactly one fold model k by a seeded, balanced random permutation (seed 2026, stratified by cohort). The assignment file is saved and committed before inference.
- **Matched baselines.** Each patient's forecasts are compared with the AR(1) and damped-persistence baselines **of the same fold k** (fitted on the same LUMIERE training pairs as that fold's flow model).
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
- **ALPTDG only:** voxel AUC of predicted uncertainty for the expert change masks (token counts as changed if ≥ 10% of its voxels lie in the mask; AUC over brain tokens per patient, then averaged).

**Descriptive only (no hypothesis tests)**

- PSNR / SSIM on brain voxels (image-space mean of 8 decoded samples vs. decoded follow-up), with Copy.
- Forecast-segmentation Dice vs. pipeline Copy (segmentation of the decoded baseline) and ceiling (segmentation of the decoded true follow-up).

---

## 7. Statistics

- **Unit of analysis: patient.** MU-Glioma-Post pairs are averaged per patient before testing.
- **H1a, H1b:** two-sided Wilcoxon signed-rank tests, flow vs. AR(1).
- **H2:** patient-bootstrap test of the difference in calibration error (5,000 resamples, seed 0).
- **Multiplicity:** Holm correction across the six primary tests (H1a, H1b, H2 × two cohorts); separately, Holm within the secondary family.
- **Effect sizes:** 95% patient-bootstrap confidence intervals of the relative improvement; rank-biserial correlation for Wilcoxon tests.
- **Every pre-specified outcome is reported, regardless of result.**

---

## 8. Pre-specified exploratory subgroups

- Histology: GBM vs. non-GBM.
- Δt inside vs. outside the LUMIERE 5th–95th percentile range.
- Time since surgery: median split (ALPTDG).
- ALPTDG expert change category (increase / decrease / unchanged / mixed).

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

## Appendix A — Frozen values and items to complete before freeze

| Item | Value |
|---|---|
| T_0 … T_4 (full-fold LUMIERE fit) | 1.35, 1.35, 1.20, 1.40, 1.40 (`runs/FROZEN_T_full_fold.json`) |
| `FROZEN_BASELINES.pt` SHA-256 | `56ec59135e54d190772e8880bfca09f884361a0668fb3e4999574f7195c8c347` |
| Segmentation-head SHA-256 | _from `runs/eval_unified/flow/meta.json` → `seg_head.sha256`_ |
| SRI24 template source and version | _to complete_ |
| MNI152 reference (FSL template or LUMIERE mean template) | _to complete_ |
| ALPTDG label integer mapping (raw → unified) | _to complete_ |
| MU-Glioma-Post label integer mapping (raw → unified) | _to complete_ |
| MU-Glioma-Post interval field (column name / derivation) | _to complete_ |
| DDPM external evaluation included? | No (proposed) — confirm |
| Patient-to-model assignment file (path + SHA-256) | _to complete_ |
| External-mode evaluation code commit | _to complete_ |

## Appendix B — Freeze checklist

- [ ] Appendix A complete
- [ ] `runs/FROZEN_MODELS.json`, `runs/FROZEN_BASELINES.pt`, `runs/FROZEN_T_full_fold.json` committed
- [ ] Evaluation code (external mode) committed
- [ ] Patient-to-model assignment file committed
- [ ] Freeze commit hash recorded in the header table
- [ ] Only then: run inference
