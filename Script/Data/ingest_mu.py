#!/usr/bin/env python3
"""
ingest_mu.py — MU-Glioma-Post -> LUMIERE grid -> frozen-VQGAN latents.

Pipeline QC only (allowed before the plan is frozen): no forecasting model is used.

Steps
  1. Inventory: complete timepoints (4 contrasts + tumorMask), days from the clinical table.
  2. Templates: LUMIERE reference = mean skull-stripped T1 (first complete timepoint per
     patient; already on the MNI152 1 mm grid). MU template = mean T1n (first complete
     timepoint per patient; native SRI24/BraTS grid). Each image is scaled by its brain
     mean before averaging; tumors wash out across patients.
  3. ONE affine (12 DOF, ANTs) MU template -> LUMIERE reference, applied identically to
     every volume of every timepoint (images: linear; labels: nearest neighbour).
  4. HARD CHECK on the first output: shape (182,218,182), affine == LUMIERE affine, and
     the in-memory array equals the written file. Any mismatch stops the script.
  5. Per-patient QC: brain-mask Dice vs the reference mask >= 0.90; patients below the
     gate get a per-patient affine (first timepoint T1n -> reference), logged, re-checked.
  6. Labels: harmonize(..., "brats2024")  ->  {0,1 NETC,2 SNFH,3 ET}; RC folded into 0.
     Semantics check on the first 10 timepoints (ET brightest on T1c when present).
  7. Unchanged transforms.preprocess (per-scan z-score) -> frozen VQGAN encoder -> cache.
     VQGAN round-trip PSNR (whole volume and brain voxels) on 6 timepoints.
  8. Pair index with flags (dt known, dt >= 7 d, baseline before surgery, diagnosis,
     grade, registration QC) — eligibility is applied at evaluation time, per the plan.

Outputs
  latent_cache_mu/MU-<pid>_tp<k>.pt     {'z','seg','mask','patient','tp','day',...}
  external/mu/mu_pairs.json             all consecutive pairs with flags
  external/mu/ingest_summary.json       counts, QC, gates
  external/mu/qc_overlays.png           20 random patients, registered T1c + reference mask
  external/mu/mu_to_lumiere_*.mat       the cohort transform (+ per-patient fallbacks)

Run:  CUDA_VISIBLE_DEVICES=0 python ingest_mu.py         (CPU-heavy registration, ~1 h)
Resumable: timepoints already cached are skipped.
"""
from __future__ import annotations
import os, sys, re, glob, json, time, shutil
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
import cache_latents as CL                                   # LUMIERE_ROOT, VQGAN_CKPT, load_vqgan, DEVICE
from lumiere_dataset import discover, CHANNEL_ORDER
from harmonize_labels import harmonize, ET
from transforms import preprocess
from vqgan3d import compute_psnr

MU_DIR = os.path.join(HERE, "mu-glioma-post")
MU_ROOT = os.path.join(MU_DIR, "MU-Glioma-Post")
CLIN = os.path.join(MU_DIR, "MU-Glioma-Post_ClinicalData-July2025.xlsx")
WORK = os.path.join(HERE, "external", "mu")
CACHE = os.path.join(HERE, "latent_cache_mu")
SUFFIX = {"T1pre": "brain_t1n", "T1post": "brain_t1c", "FLAIR": "brain_t2f", "T2": "brain_t2w"}
LUM_SHAPE = (182, 218, 182)
LUM_AFFINE = np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], float)
DICE_GATE, N_QC_PNG, N_GATE, N_SEM, SEED = 0.90, 20, 6, 10, 2026
ORD = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
DAYCOL = {k: f"Number of Days from Diagnosis to {ORD[k - 1]} MRI (Timepoint_{k})" for k in range(1, 7)}
SURG = "Number of days from Diagnosis to First surgery or procedure"
assert list(SUFFIX) == list(CHANNEL_ORDER), f"channel order mismatch: {list(SUFFIX)} vs {CHANNEL_ORDER}"


def mu_path(pid, tp, suffix):
    return os.path.join(MU_ROOT, f"PatientID_{pid:04d}", f"Timepoint_{tp}",
                        f"PatientID_{pid:04d}_Timepoint_{tp}_{suffix}.nii.gz")


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 1.0


# ───────────────────────── 1. inventory ─────────────────────────
def inventory():
    clin = pd.read_excel(CLIN, sheet_name="MU Glioma Post")
    clin.columns = [str(c).strip() for c in clin.columns]
    clin["pid"] = clin["Patient_ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    C = clin.set_index("pid")
    rows = []
    for d in sorted(glob.glob(os.path.join(MU_ROOT, "PatientID_*", "Timepoint_*"))):
        m = re.search(r"PatientID_(\d+)[/\\]Timepoint_(\d+)", d)
        pid, tp = int(m.group(1)), int(m.group(2))
        complete = all(os.path.exists(mu_path(pid, tp, s)) for s in list(SUFFIX.values()) + ["tumorMask"])
        num = lambda col: pd.to_numeric(C.at[pid, col], errors="coerce") if pid in C.index and col in C.columns else np.nan
        rows.append(dict(pid=pid, tp=tp, complete=complete, day=num(DAYCOL.get(tp, "")), surg=num(SURG),
                         diagnosis=str(C.at[pid, "Primary Diagnosis"]) if pid in C.index else "",
                         grade=str(C.at[pid, "Grade of Primary Brain Tumor"]) if pid in C.index else ""))
    return pd.DataFrame(rows)


# ───────────────────────── 2-3. templates and cohort transform ─────────────────────────
def mean_template(paths, out):
    acc, n, aff = None, 0, None
    for p in paths:
        img = nib.load(p)
        a = np.asanyarray(img.dataobj).astype(np.float64)
        nz = a > 0
        if nz.sum() < 1000:
            continue
        a = np.where(nz, a / a[nz].mean(), 0.0)
        acc = a if acc is None else acc + a
        n, aff = n + 1, img.affine
    nib.save(nib.Nifti1Image((acc / n).astype(np.float32), aff), out)
    return n


def build_templates(T):
    ref_path, mu_path_t = os.path.join(WORK, "lumiere_ref_T1.nii.gz"), os.path.join(WORK, "mu_template_T1n.nii.gz")
    if not os.path.exists(ref_path):
        pats = discover(CL.LUMIERE_ROOT)
        firsts = [next(t for t in tps if t.complete()).contrast("T1pre") for tps in pats.values()
                  if any(t.complete() for t in tps)]
        n = mean_template(firsts, ref_path)
        print(f"[mu] LUMIERE reference template from {n} patients -> {ref_path}")
    if not os.path.exists(mu_path_t):
        firsts = [mu_path(pid, int(g[g.complete].tp.min()), SUFFIX["T1pre"])
                  for pid, g in T.groupby("pid") if g.complete.any()]
        n = mean_template(firsts, mu_path_t)
        print(f"[mu] MU template from {n} patients -> {mu_path_t}")
    ref_img = nib.load(ref_path)
    assert ref_img.shape == LUM_SHAPE and np.allclose(ref_img.affine, LUM_AFFINE, atol=1e-3), \
        "reference template is not on the LUMIERE grid"
    return ref_path, mu_path_t


def affine_reg(fixed, moving, tag):
    kw = dict(fixed=fixed, moving=moving, type_of_transform="Affine")
    try:
        reg = ants.registration(**kw, random_seed=SEED)
    except TypeError:
        reg = ants.registration(**kw)
    keep = []
    for i, f in enumerate(reg["fwdtransforms"]):
        dst = os.path.join(WORK, f"mu_to_lumiere_{tag}_{i}{os.path.splitext(f)[1]}")
        shutil.copy(f, dst)
        keep.append(dst)
    return keep


def warp(fixed, path, fwd, interp):
    return ants.apply_transforms(fixed=fixed, moving=ants.image_read(path), transformlist=fwd,
                                 interpolator=interp)


# ───────────────────────── main ─────────────────────────
@torch.no_grad()
def main():
    os.makedirs(WORK, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    T = inventory()
    print(f"[mu] patients {T.pid.nunique()} | timepoints {len(T)} | complete {int(T.complete.sum())}")
    ref_path, mu_tpl = build_templates(T)
    ref = ants.image_read(ref_path)
    ref_mask = ref.numpy() > 0.3
    tpl_fwd = os.path.join(WORK, "mu_to_lumiere_cohort_0.mat")
    if os.path.exists(tpl_fwd):
        cohort_fwd = [tpl_fwd]
    else:
        cohort_fwd = affine_reg(ref, ants.image_read(mu_tpl), "cohort")
    tpl_w = warp(ref, mu_tpl, cohort_fwd, "linear").numpy()
    tpl_dice = dice(tpl_w > 0.3, ref_mask)
    print(f"[mu] cohort transform: template brain-mask Dice vs reference = {tpl_dice:.3f}")

    # hard check on the first complete timepoint
    first = T[T.complete].sort_values(["pid", "tp"]).iloc[0]
    probe = warp(ref, mu_path(int(first.pid), int(first.tp), SUFFIX["T1post"]), cohort_fwd, "linear")
    tmp = os.path.join(WORK, "_hardcheck.nii.gz")
    ants.image_write(probe, tmp)
    chk = nib.load(tmp)
    ok = chk.shape == LUM_SHAPE and np.allclose(chk.affine, LUM_AFFINE, atol=1e-3) and \
        np.allclose(np.asanyarray(chk.dataobj), probe.numpy(), atol=1e-4)
    print(f"[mu] HARD CHECK (shape, affine, array order): {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"     shape {chk.shape}\n     affine\n{np.round(chk.affine, 3)}")
        raise SystemExit("[mu] geometry hard check failed — stopping.")

    vq = CL.load_vqgan(CL.VQGAN_CKPT)
    rng = np.random.RandomState(SEED)
    pids = sorted(T[T.complete].pid.unique())
    qc_pids = set(rng.choice(pids, size=min(N_QC_PNG, len(pids)), replace=False).tolist())
    qc_slices, reg_log, sem, gate, t0, n_done = {}, {}, [], [], time.time(), 0

    for pid in pids:
        tps = sorted(T[(T.pid == pid) & T.complete].tp.astype(int))
        fwd, method = cohort_fwd, "cohort"
        b0 = warp(ref, mu_path(pid, tps[0], SUFFIX["T1post"]), fwd, "linear").numpy() > 0
        d = dice(b0, ref_mask)
        if d < DICE_GATE:
            pat_fwd = affine_reg(ref, ants.image_read(mu_path(pid, tps[0], SUFFIX["T1pre"])), f"p{pid:04d}")
            b1 = warp(ref, mu_path(pid, tps[0], SUFFIX["T1post"]), pat_fwd, "linear").numpy() > 0
            d1 = dice(b1, ref_mask)
            if d1 > d:
                fwd, method, d = pat_fwd, "per_patient", d1
        reg_log[int(pid)] = dict(method=method, dice=round(d, 4), passed=bool(d >= DICE_GATE))

        for tp in tps:
            out = os.path.join(CACHE, f"MU-{pid:04d}_tp{tp}.pt")
            if os.path.exists(out):
                n_done += 1
                continue
            img = np.stack([warp(ref, mu_path(pid, tp, SUFFIX[c]), fwd, "linear").numpy()
                            for c in CHANNEL_ORDER]).astype(np.float32)
            seg = np.rint(warp(ref, mu_path(pid, tp, "tumorMask"), fwd, "nearestNeighbor").numpy()).astype(np.int64)
            seg = harmonize(seg, "brats2024")                              # RC -> background
            mask = (img != 0).any(0)
            if len(sem) < N_SEM:
                t1c = img[CHANNEL_ORDER.index("T1post")]
                means = {int(k): float(t1c[seg == k].mean()) for k in np.unique(seg) if k != 0}
                if ET in means:
                    sem.append(dict(pid=int(pid), tp=tp, means=means, et_brightest=max(means, key=means.get) == ET))
            pp = preprocess(torch.from_numpy(img), torch.from_numpy(seg), torch.from_numpy(mask))
            pp = {k: (v.as_tensor() if hasattr(v, "as_tensor") else v) for k, v in pp.items()}
            x = pp["image"].unsqueeze(0).to(CL.DEVICE)
            z = vq.encode(x)
            if len(gate) < N_GATE:
                xr = vq.decode(z)
                m = pp["mask"].to(CL.DEVICE)
                L = (x[0][:, m].max() - x[0][:, m].min()).clamp_min(1e-6)
                mse_b = ((xr[0] - x[0]) ** 2)[:, m].mean().clamp_min(1e-12)
                gate.append(dict(whole=float(compute_psnr(x, xr)), brain=float(10 * torch.log10(L * L / mse_b))))
            day = T[(T.pid == pid) & (T.tp == tp)].day.iloc[0]
            torch.save({"z": z.squeeze(0).to(torch.float16).cpu(), "seg": pp["seg"].to(torch.uint8),
                        "mask": pp["mask"], "patient": f"MU-{pid:04d}", "tp": int(tp),
                        "day": None if pd.isna(day) else float(day), "cohort": "mu",
                        "norm": "per_scan", "registration": method}, out)
            if pid in qc_pids and tp == tps[0]:
                k = int(np.argmax(ref_mask.sum((0, 1))))
                qc_slices[int(pid)] = (img[CHANNEL_ORDER.index("T1post")][:, :, k], ref_mask[:, :, k])
            n_done += 1
            if n_done % 25 == 0:
                print(f"  cached {n_done}/{int(T.complete.sum())}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # pair index with flags
    pairs = []
    for pid, g in T[T.complete].sort_values("tp").groupby("pid"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            dt = b.day - a.day
            q = reg_log.get(int(pid), {})
            pairs.append(dict(patient=f"MU-{int(pid):04d}", tp_a=int(a.tp), tp_b=int(b.tp),
                              dt=None if pd.isna(dt) else float(dt),
                              baseline_pre_surgery=bool(a.day < a.surg) if not (pd.isna(a.day) or pd.isna(a.surg)) else False,
                              diagnosis=a.diagnosis, grade=a.grade,
                              registration=q.get("method"), reg_dice=q.get("dice"), reg_pass=q.get("passed"),
                              usable_basic=bool((not pd.isna(dt)) and dt >= 7 and q.get("passed", False)
                                                and not (a.day < a.surg))))
    with open(os.path.join(WORK, "mu_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=1)

    # QC overlays
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(qc_slices); cols = 5; rows_ = int(np.ceil(n / cols))
        fig, ax = plt.subplots(rows_, cols, figsize=(cols * 3, rows_ * 3.4))
        for a_, (pid, (sl, mk)) in zip(np.ravel(ax), sorted(qc_slices.items())):
            a_.imshow(np.rot90(sl), cmap="gray"); a_.contour(np.rot90(mk), levels=[0.5], colors="r", linewidths=0.8)
            a_.set_title(f"MU-{pid:04d} ({reg_log[pid]['method']}, Dice {reg_log[pid]['dice']:.2f})", fontsize=8)
            a_.axis("off")
        for a_ in np.ravel(ax)[n:]:
            a_.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(WORK, "qc_overlays.png"), dpi=110); plt.close(fig)
    except Exception as e:
        print(f"[mu] QC overlay figure skipped: {e}")

    dices = np.array([v["dice"] for v in reg_log.values()])
    summ = dict(patients=int(T.pid.nunique()), timepoints=int(len(T)), complete=int(T.complete.sum()),
                cached=n_done, template_dice=tpl_dice, hard_check="PASS",
                reg_dice=dict(median=float(np.median(dices)), min=float(dices.min()),
                              below_gate=int((dices < DICE_GATE).sum()),
                              per_patient_fallbacks=int(sum(v["method"] == "per_patient" for v in reg_log.values()))),
                semantics=dict(checked=len(sem), et_brightest=int(sum(s["et_brightest"] for s in sem)), cases=sem),
                vqgan_gate=dict(whole_mean=float(np.mean([g["whole"] for g in gate])) if gate else None,
                                brain_mean=float(np.mean([g["brain"] for g in gate])) if gate else None),
                pairs=dict(total=len(pairs), usable_basic=int(sum(p["usable_basic"] for p in pairs)),
                           usable_patients=len({p["patient"] for p in pairs if p["usable_basic"]})),
                registration_log=reg_log)
    with open(os.path.join(WORK, "ingest_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("=" * 78)
    print(f"  MU-Glioma-Post ingestion — cached {n_done} timepoints in {(time.time() - t0) / 60:.0f} min")
    print(f"  template Dice {tpl_dice:.3f} | per-patient Dice median {summ['reg_dice']['median']:.3f}, "
          f"min {summ['reg_dice']['min']:.3f}, below {DICE_GATE}: {summ['reg_dice']['below_gate']} "
          f"(fallbacks used: {summ['reg_dice']['per_patient_fallbacks']})")
    print(f"  label semantics: ET brightest on T1c in {summ['semantics']['et_brightest']}/{summ['semantics']['checked']} checked timepoints")
    if gate:
        print(f"  VQGAN round-trip PSNR: whole {summ['vqgan_gate']['whole_mean']:.2f} dB | "
              f"brain {summ['vqgan_gate']['brain_mean']:.2f} dB  (LUMIERE: whole 33.31, brain 26.99)")
    print(f"  pairs: {summ['pairs']['total']} total | usable (dt>=7, post-surgery, reg QC pass) "
          f"{summ['pairs']['usable_basic']} from {summ['pairs']['usable_patients']} patients")
    print(f"  outputs: {WORK}/ (mu_pairs.json, ingest_summary.json, qc_overlays.png) and {CACHE}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
