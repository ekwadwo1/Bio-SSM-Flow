#!/usr/bin/env python3
"""
ingest_alptdg.py — UCSF-ALPTDG -> LUMIERE grid -> frozen-VQGAN latents.

Pipeline QC only (allowed before the plan is frozen): no forecasting model is used.
Same design as ingest_mu.py, with its OWN cohort template and transform (the ALPTDG
headers use a different origin convention from MU-Glioma-Post).

Steps
  1. Inventory: 298 patients x {time1, time2} x {t1, t1ce, t2, flair, seg} + change annotations.
  2. ALPTDG template = mean time-1 T1 (brain-mean scaled); reference = the LUMIERE mean
     T1 already built by ingest_mu.py (external/mu/lumiere_ref_T1.nii.gz).
  3. ONE affine (12 DOF) template -> reference, applied to both timepoints of every patient;
     per-patient fallback (time-1 T1 -> reference) for patients below the Dice gate.
  4. HARD CHECK on the first output (shape, affine, array order) — stops on failure.
  5. Left-right orientation check (template as-is vs mirrored).
  6. Labels: harmonize(..., "brats2024") (1 NCR/NETC, 2 SNFH, 3 ET, 4 RC -> 0); semantics
     check on 10 timepoints. Change annotations warped (nearest neighbour), resized to 128^3,
     cached per patient; their 1/2 coding is checked against Table S1 volumes (10 patients).
  7. Unchanged transforms.preprocess -> frozen VQGAN encoder -> cache; round-trip PSNR on 6.
  8. Pair index (one pair per patient) with flags: dt, time since surgery (raw value),
     intervening surgery, WHO 2021 diagnosis, grade, UCSF-PDGM overlap, registration QC.

Outputs
  latent_cache_alptdg/ALPTDG-<pid>_t<k>.pt   {'z','seg','mask','patient','tp','day',...}
  latent_cache_alptdg/ALPTDG-<pid>_change.pt {'flair','t1ce'} uint8 [128^3] (0/1/2 as released)
  external/alptdg/alptdg_pairs.json, ingest_summary.json, qc_overlays*.png, transforms

Run:  CUDA_VISIBLE_DEVICES=0 python ingest_alptdg.py      (~30-40 min; resumable)
"""
from __future__ import annotations
import os, sys, json, time, shutil
import numpy as np
import pandas as pd
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import ants
except ImportError:
    raise SystemExit("ANTsPy is required: pip install antspyx")
import cache_latents as CL
import ingest_mu as IM                                      # mean_template, dice, warp, LUM_* constants
from lumiere_dataset import CHANNEL_ORDER
from harmonize_labels import harmonize, ET
from transforms import preprocess, _resize_label
from vqgan3d import compute_psnr

ROOT = os.path.join(HERE, "ucsf-alptdg")
TABLE = os.path.join(ROOT, "UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx")
WORK = os.path.join(HERE, "external", "alptdg")
CACHE = os.path.join(HERE, "latent_cache_alptdg")
REF = os.path.join(HERE, "external", "mu", "lumiere_ref_T1.nii.gz")
SUFFIX = {"T1pre": "t1", "T1post": "t1ce", "FLAIR": "flair", "T2": "t2"}
CHANGE = {"flair": "flair_subtraction_seg", "t1ce": "t1ce_subtraction_seg"}
N_QC_PNG, N_LOW, N_GATE, N_SEM, N_CODE, SEED = 20, 10, 6, 10, 10, 2026
assert list(SUFFIX) == list(CHANNEL_ORDER)


def tp_path(pid, k, suffix):
    return os.path.join(ROOT, pid, f"{pid}_time{k}_{suffix}.nii.gz")


def ch_path(pid, key):
    return os.path.join(ROOT, pid, f"{pid}_{CHANGE[key]}.nii.gz")


def norm_cols(df):
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


def load_tables():
    xl = pd.ExcelFile(TABLE)
    ci = norm_cols(xl.parse("Clinical Info"))
    ci["pid"] = ci["SubjectID"].astype(str).str.extract(r"(\d+)")[0]
    pdgm = xl.parse("UCSF-PDGM Mapping")
    pdgm_ids = set(pdgm.iloc[:, 0].astype(str).str.extract(r"(\d+)")[0].dropna())
    vol = norm_cols(xl.parse("UCSF-LPTDG"))
    vol["pid"] = vol["SubjectID"].astype(str).str.extract(r"(\d+)")[0]
    return ci.set_index("pid"), pdgm_ids, vol


def affine_reg(fixed, moving, tag):
    kw = dict(fixed=fixed, moving=moving, type_of_transform="Affine")
    try:
        reg = ants.registration(**kw, random_seed=SEED)
    except TypeError:
        reg = ants.registration(**kw)
    out = []
    for i, f in enumerate(reg["fwdtransforms"]):
        dst = os.path.join(WORK, f"alptdg_to_lumiere_{tag}_{i}{os.path.splitext(f)[1]}")
        shutil.copy(f, dst)
        out.append(dst)
    return out, reg


def overlay(path, items, ref_mask, title):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = int(np.argmax(ref_mask.sum((0, 1))))
        cols = 5; rows = int(np.ceil(len(items) / cols))
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
        for a, (lab, img) in zip(np.ravel(ax), items):
            a.imshow(np.rot90(img[:, :, k]), cmap="gray")
            a.contour(np.rot90(ref_mask[:, :, k]), levels=[0.5], colors="r", linewidths=0.8)
            a.set_title(lab, fontsize=8); a.axis("off")
        for a in np.ravel(ax)[len(items):]:
            a.axis("off")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[alptdg] overlay {path} skipped: {e}")


@torch.no_grad()
def main():
    os.makedirs(WORK, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(REF):
        raise SystemExit(f"[alptdg] reference template missing: {REF} (run ingest_mu.py first)")
    ci, pdgm_ids, vol = load_tables()
    pids = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and d.isdigit())
    complete = {p: all(os.path.exists(tp_path(p, k, s)) for k in (1, 2) for s in list(SUFFIX.values()) + ["seg"])
                for p in pids}
    pids_c = [p for p in pids if complete[p]]
    print(f"[alptdg] patients {len(pids)} | complete (both timepoints, 4 contrasts + seg) {len(pids_c)}")

    # template + cohort transform
    tpl = os.path.join(WORK, "alptdg_template_T1.nii.gz")
    if not os.path.exists(tpl):
        n = IM.mean_template([tp_path(p, 1, SUFFIX["T1pre"]) for p in pids_c], tpl)
        print(f"[alptdg] ALPTDG template from {n} patients -> {tpl}")
    ref = ants.image_read(REF)
    ref_mask = ref.numpy() > 0.3
    cohort_fwd, _ = affine_reg(ref, ants.image_read(tpl), "cohort")
    tpl_dice = IM.dice(IM.warp(ref, tpl, cohort_fwd, "linear").numpy() > 0.3, ref_mask)
    print(f"[alptdg] cohort transform: template brain-mask Dice vs reference = {tpl_dice:.3f}")

    # hard check
    probe = IM.warp(ref, tp_path(pids_c[0], 1, SUFFIX["T1post"]), cohort_fwd, "linear")
    tmp = os.path.join(WORK, "_hardcheck.nii.gz"); ants.image_write(probe, tmp)
    chk = nib.load(tmp)
    ok = chk.shape == IM.LUM_SHAPE and np.allclose(chk.affine, IM.LUM_AFFINE, atol=1e-3) and \
        np.allclose(np.asanyarray(chk.dataobj), probe.numpy(), atol=1e-4)
    print(f"[alptdg] HARD CHECK (shape, affine, array order): {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("[alptdg] geometry hard check failed — stopping.")

    # left-right orientation check
    t_img = ants.image_read(tpl)
    mir = ants.from_numpy(t_img.numpy()[::-1].copy(), origin=t_img.origin, spacing=t_img.spacing,
                          direction=t_img.direction)
    r = ref.numpy(); m = r > 0.3
    corr = {}
    for name, img in (("as_is", t_img), ("mirrored", mir)):
        w = ants.registration(fixed=ref, moving=img, type_of_transform="Affine")["warpedmovout"].numpy()
        corr[name] = float(np.corrcoef(w[m], r[m])[0, 1])
    print(f"[alptdg] orientation check: as-is {corr['as_is']:.4f} vs mirrored {corr['mirrored']:.4f} "
          f"-> {'OK' if corr['as_is'] >= corr['mirrored'] else 'INVESTIGATE'}")

    # change-annotation coding vs Table S1 (native space, 1 mm^3 voxels)
    code_rows = []
    for p in pids_c:
        if len(code_rows) >= N_CODE:
            break
        fl = np.asanyarray(nib.load(ch_path(p, "flair")).dataobj) if os.path.exists(ch_path(p, "flair")) else None
        if fl is None or not (fl > 0).any():
            continue
        tv = vol[vol.pid == p]
        pick = lambda col: tv[col].dropna().iloc[0] if col in tv and tv[col].notna().any() else None
        code_rows.append(dict(pid=p, flair_label1=int((fl == 1).sum()), flair_label2=int((fl == 2).sum()),
                              SNFH_Inc_Volume=pick("SNFH_Inc_Volume"), SNFH_Dec_Volume=pick("SNFH_Dec_Volume")))
    print("[alptdg] change-annotation coding check (voxel counts vs Table S1 volumes):")
    for row in code_rows:
        print(f"   {row}")

    vq = CL.load_vqgan(CL.VQGAN_CKPT)
    rng = np.random.RandomState(SEED)
    qc_pids = set(rng.choice(pids_c, size=min(N_QC_PNG, len(pids_c)), replace=False).tolist())
    reg_log, sem, gate, qc_items, t0, n_done = {}, [], [], [], time.time(), 0
    first_t1c = {}
    for p in pids_c:
        fwd, method = cohort_fwd, "cohort"
        b0 = IM.warp(ref, tp_path(p, 1, SUFFIX["T1post"]), fwd, "linear").numpy()
        d = IM.dice(b0 > 0, ref_mask)
        if d < IM.DICE_GATE:
            pf, _ = affine_reg(ref, ants.image_read(tp_path(p, 1, SUFFIX["T1pre"])), f"p{p}")
            b1 = IM.warp(ref, tp_path(p, 1, SUFFIX["T1post"]), pf, "linear").numpy()
            d1 = IM.dice(b1 > 0, ref_mask)
            if d1 > d:
                fwd, method, d, b0 = pf, "per_patient", d1, b1
        reg_log[p] = dict(method=method, dice=round(d, 4), passed=bool(d >= IM.DICE_GATE), fwd=fwd)
        first_t1c[p] = b0[:, :, int(np.argmax(ref_mask.sum((0, 1))))]
        if p in qc_pids:
            qc_items.append((f"ALPTDG-{p} ({method}, Dice {d:.2f})", b0))

        for k in (1, 2):
            out = os.path.join(CACHE, f"ALPTDG-{p}_t{k}.pt")
            if os.path.exists(out):
                n_done += 1
                continue
            img = np.stack([IM.warp(ref, tp_path(p, k, SUFFIX[c]), fwd, "linear").numpy()
                            for c in CHANNEL_ORDER]).astype(np.float32)
            seg = np.rint(IM.warp(ref, tp_path(p, k, "seg"), fwd, "nearestNeighbor").numpy()).astype(np.int64)
            seg = harmonize(seg, "brats2024")
            mask = (img != 0).any(0)
            if len(sem) < N_SEM:
                t1c = img[CHANNEL_ORDER.index("T1post")]
                means = {int(v): float(t1c[seg == v].mean()) for v in np.unique(seg) if v != 0}
                if ET in means:
                    sem.append(dict(pid=p, tp=k, means=means, et_brightest=max(means, key=means.get) == ET))
            pp = preprocess(torch.from_numpy(img), torch.from_numpy(seg), torch.from_numpy(mask))
            pp = {kk: (v.as_tensor() if hasattr(v, "as_tensor") else v) for kk, v in pp.items()}
            x = pp["image"].unsqueeze(0).to(CL.DEVICE)
            z = vq.encode(x)
            if len(gate) < N_GATE:
                xr = vq.decode(z)
                mm = pp["mask"].to(CL.DEVICE)
                L = (x[0][:, mm].max() - x[0][:, mm].min()).clamp_min(1e-6)
                mse_b = ((xr[0] - x[0]) ** 2)[:, mm].mean().clamp_min(1e-12)
                gate.append(dict(whole=float(compute_psnr(x, xr)), brain=float(10 * torch.log10(L * L / mse_b))))
            dt = pd.to_numeric(ci.at[p, "Days from 1st scan to 2nd scan"], errors="coerce") if p in ci.index else np.nan
            day = 0.0 if k == 1 else (None if pd.isna(dt) else float(dt))
            torch.save({"z": z.squeeze(0).to(torch.float16).cpu(), "seg": pp["seg"].to(torch.uint8),
                        "mask": pp["mask"], "patient": f"ALPTDG-{p}", "tp": k, "day": day,
                        "cohort": "alptdg", "norm": "per_scan", "registration": method}, out)
            n_done += 1
        cpath = os.path.join(CACHE, f"ALPTDG-{p}_change.pt")
        if not os.path.exists(cpath):
            ch = {}
            for key in CHANGE:
                if os.path.exists(ch_path(p, key)):
                    a = np.rint(IM.warp(ref, ch_path(p, key), fwd, "nearestNeighbor").numpy()).astype(np.int64)
                    rz = _resize_label(torch.from_numpy(a))
                    rz = rz.as_tensor() if hasattr(rz, "as_tensor") else rz
                    ch[key] = rz.to(torch.uint8)
            torch.save(ch, cpath)
        if n_done % 50 == 0:
            print(f"  cached {n_done}/{2 * len(pids_c)} timepoints ({(time.time() - t0) / 60:.1f} min)", flush=True)

    low = sorted(reg_log.items(), key=lambda kv: kv[1]["dice"])[:N_LOW]
    low_items = []
    for p, v in low:
        img = IM.warp(ref, tp_path(p, 1, SUFFIX["T1post"]), v["fwd"], "linear").numpy()
        low_items.append((f"ALPTDG-{p} ({v['method']}, Dice {v['dice']:.3f})", img))
    overlay(os.path.join(WORK, "qc_overlays.png"), qc_items, ref_mask, "ALPTDG: 20 random patients")
    overlay(os.path.join(WORK, "qc_overlays_lowest_dice.png"), low_items, ref_mask, "ALPTDG: 10 lowest Dice")

    # pair index
    def num(p, col):
        return pd.to_numeric(ci.at[p, col], errors="coerce") if (p in ci.index and col in ci.columns) else np.nan
    pairs = []
    for p in pids_c:
        dt = num(p, "Days from 1st scan to 2nd scan")
        sx2 = num(p, "Days from 1st scan to 2nd SX")
        q = reg_log[p]
        pairs.append(dict(patient=f"ALPTDG-{p}", dt=None if pd.isna(dt) else float(dt),
                          surgery_to_scan1_raw=None if pd.isna(num(p, "Days from 1st surgery/DX to 1st scan"))
                          else float(num(p, "Days from 1st surgery/DX to 1st scan")),
                          intervening_surgery=bool((not pd.isna(sx2)) and (not pd.isna(dt)) and 0 < sx2 <= dt),
                          diagnosis=str(ci.at[p, "WHO 2021 Diagnosis"]) if p in ci.index else "",
                          grade=str(ci.at[p, "Grade"]) if p in ci.index else "",
                          pdgm_overlap=p in pdgm_ids, registration=q["method"], reg_dice=q["dice"],
                          reg_pass=q["passed"],
                          usable_basic=bool((not pd.isna(dt)) and dt >= 7 and q["passed"])))
    with open(os.path.join(WORK, "alptdg_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=1)

    dices = np.array([v["dice"] for v in reg_log.values()])
    s2s = np.array([pp_["surgery_to_scan1_raw"] for pp_ in pairs if pp_["surgery_to_scan1_raw"] is not None])
    dts = np.array([pp_["dt"] for pp_ in pairs if pp_["dt"] is not None])
    summ = dict(patients=len(pids), complete=len(pids_c), cached_timepoints=n_done, template_dice=tpl_dice,
                hard_check="PASS", orientation=corr,
                reg_dice=dict(median=float(np.median(dices)), min=float(dices.min()),
                              below_gate=int((dices < IM.DICE_GATE).sum()),
                              per_patient_fallbacks=int(sum(v["method"] == "per_patient" for v in reg_log.values()))),
                semantics=dict(checked=len(sem), et_brightest=int(sum(s["et_brightest"] for s in sem)), cases=sem),
                change_coding_check=code_rows,
                vqgan_gate=dict(whole_mean=float(np.mean([g["whole"] for g in gate])) if gate else None,
                                brain_mean=float(np.mean([g["brain"] for g in gate])) if gate else None),
                dt_days=dict(median=float(np.median(dts)), p5=float(np.percentile(dts, 5)), p95=float(np.percentile(dts, 95)),
                             below7=int((dts < 7).sum()), missing=int(len(pairs) - len(dts))),
                surgery_to_scan1_raw=dict(min=float(s2s.min()), median=float(np.median(s2s)), max=float(s2s.max()),
                                          n_negative=int((s2s < 0).sum()), n_zero=int((s2s == 0).sum())),
                pairs=dict(total=len(pairs), usable_basic=int(sum(x["usable_basic"] for x in pairs)),
                           intervening_surgery=int(sum(x["intervening_surgery"] for x in pairs)),
                           pdgm_overlap=int(sum(x["pdgm_overlap"] for x in pairs))),
                diagnosis_counts=pd.Series([x["diagnosis"] for x in pairs]).value_counts().to_dict(),
                registration_log={p: {k: v for k, v in d.items() if k != "fwd"} for p, d in reg_log.items()})
    with open(os.path.join(WORK, "ingest_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)

    print("=" * 80)
    print(f"  UCSF-ALPTDG ingestion — {n_done} timepoints cached in {(time.time() - t0) / 60:.0f} min")
    print(f"  template Dice {tpl_dice:.3f} | per-patient Dice median {summ['reg_dice']['median']:.3f}, "
          f"min {summ['reg_dice']['min']:.3f}, below {IM.DICE_GATE}: {summ['reg_dice']['below_gate']} "
          f"(fallbacks used: {summ['reg_dice']['per_patient_fallbacks']})")
    print(f"  orientation: as-is {corr['as_is']:.4f} vs mirrored {corr['mirrored']:.4f}")
    print(f"  label semantics: ET brightest on T1c in {summ['semantics']['et_brightest']}/{summ['semantics']['checked']}")
    if gate:
        print(f"  VQGAN round-trip PSNR: whole {summ['vqgan_gate']['whole_mean']:.2f} dB | brain "
              f"{summ['vqgan_gate']['brain_mean']:.2f} dB  (LUMIERE 33.31 / 26.99; MU 32.13 / 26.97)")
    d_ = summ["dt_days"]
    print(f"  dt (days): median {d_['median']:.0f}, 5-95% {d_['p5']:.0f}-{d_['p95']:.0f}, <7: {d_['below7']}, missing: {d_['missing']}")
    s_ = summ["surgery_to_scan1_raw"]
    print(f"  'Days from 1st surgery/DX to 1st scan' raw: min {s_['min']:.0f}, median {s_['median']:.0f}, "
          f"max {s_['max']:.0f}, negative {s_['n_negative']}, zero {s_['n_zero']}")
    pr = summ["pairs"]
    print(f"  pairs: {pr['total']} | usable (dt>=7, reg QC pass) {pr['usable_basic']} | intervening surgery "
          f"{pr['intervening_surgery']} | UCSF-PDGM overlap {pr['pdgm_overlap']}")
    print(f"  diagnoses: {summ['diagnosis_counts']}")
    print(f"  outputs: {WORK}/ and {CACHE}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
