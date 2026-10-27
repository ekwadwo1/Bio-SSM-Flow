# Bio-SSM-Flow — External Validation Analysis Plan

| Field | Value |
|---|---|
| Version | v0.1 (DRAFT — not yet frozen) |
| Drafted | 2026-09-29 |
| Cohorts | UCSF-ALPTDG, MU-Glioma-Post |
| Frozen on | _to be filled at freeze_ |
| Freeze commit hash | _to be filled at freeze_ |

---

## Rule 0 — Freeze before inference

No model inference on external data is performed until **this plan, the evaluation code, and `runs/FROZEN_MODELS.json` are committed** and the commit hash is recorded above.

- **Allowed before freeze (pipeline QC):** file inventory, header/affine checks, label-integer inspection, label-semantics verification, registration quality control, VQGAN encoding.
- **Not allowed before freeze:** computing any model prediction, or any metric derived from one.

---

## 1. Hypotheses (primary, tested separately in each cohort)

- **H1 — accuracy.** The model's ensemble-mean forecast has lower in-ROI latent MSE than the Copy baseline.
- **H2 — calibration transfer.** Applying the LUMIERE-fitted temperature *unchanged* lowers calibration error (brain region, measured against the finite-R ideal) relative to no calibration (T = 1).

---

## 2. Frozen components

| Component | Specification |
|---|---|
| Forecasting models | Flow V2, fold models 0–4, EMA weights |
| Model SHA-256 (prefix) | fold0 `bda53a14cc36` · fold1 `92af8aa55d8b` · fold2 `c9c9e33b8a95` · fold3 `1524d49430c8` · fold4 `ce239c4ab756` |
| Latent autoencoder | Frozen Paper-1 VQGAN (`vqgan_r4/best_vqgan.pt`) |
| Segmentation head | `runs/seghead_img_fold0/imgseg_best.pt` (hash in Appendix A) |
| Temperature per fold, T_k | Fitted on **all** fold-k LUMIERE validation patients, same objective as the cross-fitting (brain region, finite-R ideal, grid step 0.05). Values in Appendix A. |
| Sampling | R = 32 ensemble members, 20 midpoint ODE steps, per-pair seed as defined in `eval_unified.py` |

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

**Interval Δt.** The true interval is used where available; otherwise the LUMIERE median (84 days). This is justified by the pre-established finding that Δt has no measurable effect on the forecast (true vs. shuffled vs. constant Δt; whole volume and tumor ROI; all five folds). Where available, Δt is still used for subgroup analysis.

---

## 4. Preprocessing (identical to training)

1. **Target grid.** FSL MNI152 1 mm: shape 182×218×182, affine `[[-1,0,0,90],[0,1,0,-126],[0,0,1,-72]]`, orientation LAS — the exact LUMIERE grid.
2. **Spatial mapping.** A single SRI24 → MNI152 affine transform (12 DOF), computed once from the two templates, applied identically to **both timepoints of every patient**. Images: linear interpolation. Labels: nearest-neighbour. Per-patient affine refinement is used only for patients failing QC, and is logged.
3. **Hard check.** Every output volume's shape and affine must equal the LUMIERE reference exactly; otherwise the pipeline stops.
4. **Registration QC.** Brain-mask Dice against the MNI reference mask ≥ 0.90; visual overlay check on 20 randomly selected patients per cohort.
5. **Intensity and resampling.** Channel order [T1pre, T1post, FLAIR, T2]; unchanged `transforms.py` pipeline (trilinear resize to 128³, per-channel z-score within the brain); frozen VQGAN encoder; latents cached.
6. **Labels.** Each cohort's integers mapped to {0 background, 1 NETC, 2 SNFH, 3 ET}; resection cavity (RC) folded into background. Semantics verified on 10 cases per cohort (ET brightest on T1c when present).
7. **VQGAN round-trip fidelity.** PSNR/SSIM on brain voxels reported per cohort alongside LUMIERE. Reported only; not an exclusion criterion.

---

## 5. Model application

- **Patient-to-model assignment.** Each external patient is assigned to exactly one fold model by a seeded, balanced random permutation (seed 2026, stratified by cohort). The assignment file is saved and committed **before** inference.
- **Temperature.** Each prediction uses its assigned model's own T_k.
- **Baselines.** Copy (both cohorts); capped Linear (MU-Glioma-Post pairs with a preceding timepoint; extrapolation ratio capped at 1).
- **DDPM transport.** Secondary analysis with the same design. Inclusion is decided and recorded in Appendix A before freeze, based on the long-DDPM convergence run.

---

## 6. Endpoint definitions

**Tumor ROI.** Ground-truth ET ∪ NETC at baseline or follow-up, dilated by 3 voxels, pooled to latent tokens with > 10% occupancy. Sensitivity ROI: ET ∪ NETC ∪ SNFH.

**Calibration error.** Mean over nominal levels {0.1, 0.2, …, 0.9, 0.95} of |empirical coverage − finite-R ideal coverage|, over brain tokens.

**Primary endpoints**

- In-ROI latent MSE, model vs. Copy (H1).
- Calibration error, T_k vs. T = 1 (H2).

**Secondary endpoints**

- Whole-volume latent MSE and cosine vs. Copy.
- CRPS, raw and calibrated (brain and ROI).
- PSNR / SSIM on brain voxels (decoded forecast vs. decoded follow-up), vs. Copy.
- Forecast-segmentation Dice (ET, NETC, SNFH, TC, WT) vs. Copy (baseline labels) and ceiling (segmentation of the decoded true follow-up).
- TC growth-direction accuracy (change threshold: ±10% or 50 voxels).
- Error–uncertainty Spearman, token level (whole, brain, ROI).
- Case-level uncertainty: Spearman between each patient's mean ROI uncertainty and ROI error.
- **ALPTDG only:** voxel AUC of predicted uncertainty for the expert change masks. A token counts as changed if ≥ 10% of its voxels lie in the change mask; AUC computed over brain tokens per patient, then averaged.

---

## 7. Statistics

- **Unit of analysis: patient.** MU-Glioma-Post pairs are averaged per patient before testing.
- **H1:** two-sided Wilcoxon signed-rank test, model vs. Copy.
- **H2:** patient-bootstrap test of the difference in calibration error (5,000 resamples, seed 0).
- **Multiplicity:** Holm correction across the four primary tests (H1, H2 × two cohorts); separately, Holm within the secondary family.
- **Effect sizes:** reported with 95% patient-bootstrap confidence intervals (rank-biserial correlation for Wilcoxon tests; relative improvement for MSE endpoints).
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

## Appendix A — To be completed before freeze

| Item | Value |
|---|---|
| T_0 … T_4 (full-fold LUMIERE fit) | |
| Segmentation-head SHA-256 | |
| SRI24 template source and version | |
| MNI152 reference (FSL template or LUMIERE mean template) | |
| ALPTDG label integer mapping (raw → unified) | |
| MU-Glioma-Post label integer mapping (raw → unified) | |
| MU-Glioma-Post interval field (column name / derivation) | |
| DDPM external evaluation included? (yes / no, reason) | |
| Patient-to-model assignment file (path + SHA-256) | |
| `eval_unified.py` external-mode commit | |

## Appendix B — Freeze checklist

- [ ] Appendix A complete
- [ ] `runs/FROZEN_MODELS.json` committed
- [ ] Evaluation code (external mode) committed
- [ ] Patient-to-model assignment file committed
- [ ] Freeze commit hash recorded in the header table
- [ ] Only then: run inference
