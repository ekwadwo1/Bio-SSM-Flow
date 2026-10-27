#!/usr/bin/env python3
"""
eval_external.py — pre-registered external validation (EXTERNAL_VALIDATION_PLAN v0.4).

Modes (run in this order):
  --assign                     Eligibility + patient-to-model assignment. No model output is
                               computed (allowed before the freeze). Creates
                               runs/external_eval/assignment.json once; never overwrites it.
  --smoke                      Full pipeline on a few LUMIERE validation pairs treated as an
                               "external" cohort (each scored by its own fold model). Allowed
                               before the freeze; exercises every code path.
  --cohort mu|alptdg --plan_commit <hash>
                               The real external evaluation. Refuses to run without the commit
                               hash of the frozen plan (Rule 0). Resumable.
  --summarize --cohorts ...    Statistics from the saved per-pair records.

Safeguards: frozen flow models verified via FROZEN_MODELS.json (eval_unified.load_model);
FROZEN_BASELINES.pt and the segmentation head verified against their recorded SHA-256;
eligibility checked against the counts and patient IDs written in the plan.

Per pair: frozen flow model of the assigned fold (R=32, 20 midpoint steps, full-fold T_k);
AR(1) and damped persistence of the SAME fold (Gaussian, T=1); Copy; capped Linear (MU only).
Endpoints: latent MSE / cosine (whole, brain, tumor ROI, ROI incl. SNFH); CRPS raw/calibrated
(brain, ROI); coverage curves (brain); error-uncertainty Spearman; ROI uncertainty for the
case-level analysis; ALPTDG change-annotation AUC; descriptive PSNR/SSIM and Dice.
"""
from __future__ import annotations
import os, sys, json, time, zlib, argparse, datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_unified as EU
import aggregate_unified as AG
from cached_pair_dataset import CachedPairDataset, patient_kfold
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet

DEV = EU.DEV
OUT = os.path.join(HERE, "runs", "external_eval")
ASSIGN = os.path.join(OUT, "assignment.json")
BASELINES = os.path.join(HERE, "runs", "FROZEN_BASELINES.pt")
TFILE = os.path.join(HERE, "runs", "FROZEN_T_full_fold.json")
EXPECTED = dict(baselines="56ec59135e54d190772e8880bfca09f884361a0668fb3e4999574f7195c8c347",
                seg_head="15ba42d79e0d2711f686c2aca30272d68fd28e5533a6ea9a85f605e5b53cb5d0")
COHORT = {
    "mu": dict(cache=os.path.join(HERE, "latent_cache_mu"),
               pairs=os.path.join(HERE, "external", "mu", "mu_pairs.json")),
    "alptdg": dict(cache=os.path.join(HERE, "latent_cache_alptdg"),
                   pairs=os.path.join(HERE, "external", "alptdg", "alptdg_pairs.json"),
                   table=os.path.join(HERE, "ucsf-alptdg", "UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx")),
}
PLAN = dict(alptdg_nondiffuse={"100011", "100027", "100153", "100190"},
            alptdg_presurgery={"100088", "100294"}, alptdg_eligible=292, mu_pilocytic_patients=3)
DT_FALLBACK, DT_RANGE, SEED, K, K_DECODE = 84.0, (7.0, 175.0), 2026, 5, 8
NOM = EU.NOMINAL
SEG_EVAL = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}


# ───────────────────────────── eligibility & assignment ─────────────────────────────
def norm_cols(df):
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


def base_rec(cohort, patient, tp_a, tp_b, fa, fb, dt, **kw):
    r = dict(cohort=cohort, patient=patient, tp_a=tp_a, tp_b=tp_b, file_a=fa, file_b=fb, dt=dt,
             reg_dice=None, diagnosis="", gbm=False, intervening_surgery=False, pdgm_overlap=False,
             s2s=None, change_file=None, change_snfh=None, change_et=None, file_prev=None)
    r.update(kw)
    return r


def eligible_pairs(cohort):
    cfg = COHORT[cohort]
    P = json.load(open(cfg["pairs"]))
    out = []
    if cohort == "mu":
        pil = {p["patient"] for p in P if "pilocytic" in str(p["diagnosis"]).lower()}
        if len(pil) != PLAN["mu_pilocytic_patients"]:
            raise SystemExit(f"[elig] MU: expected {PLAN['mu_pilocytic_patients']} pilocytic patients, found {len(pil)}")
        cached = {}
        for f in os.listdir(cfg["cache"]):
            if f.endswith(".pt") and "_tp" in f:
                pat, tp = f[:-3].split("_tp")
                cached.setdefault(pat, []).append(int(tp))
        for p in P:
            fa = os.path.join(cfg["cache"], f"{p['patient']}_tp{p['tp_a']}.pt")
            fb = os.path.join(cfg["cache"], f"{p['patient']}_tp{p['tp_b']}.pt")
            if not (p["usable_basic"] and p["patient"] not in pil and os.path.exists(fa) and os.path.exists(fb)):
                continue
            prev = [t for t in cached.get(p["patient"], []) if t < p["tp_a"]]
            fp = os.path.join(cfg["cache"], f"{p['patient']}_tp{max(prev)}.pt") if prev else None
            diag = str(p["diagnosis"]).strip()
            out.append(base_rec(cohort, p["patient"], p["tp_a"], p["tp_b"], fa, fb, p["dt"],
                                reg_dice=p["reg_dice"], diagnosis=diag, gbm=diag.upper() == "GBM", file_prev=fp))
        log = dict(pairs_in_index=len(P), excluded_pilocytic_patients=sorted(pil))
    else:
        xl = pd.ExcelFile(cfg["table"])
        ci = norm_cols(xl.parse("Clinical Info"))
        ci["pid"] = ci["SubjectID"].astype(str).str.extract(r"(\d+)")[0]
        pres = dict(zip(ci["pid"], ci["Presumed Diagnosis"].astype(str)))
        nondiff = {p for p, d in pres.items() if any(k in d.lower() for k in ("pilocytic", "xanthoastrocytoma"))}
        if nondiff != PLAN["alptdg_nondiffuse"]:
            raise SystemExit(f"[elig] ALPTDG non-diffuse set {sorted(nondiff)} != plan {sorted(PLAN['alptdg_nondiffuse'])}")
        vol = norm_cols(xl.parse("UCSF-LPTDG"))
        vol["pid"] = vol["SubjectID"].astype(str).str.extract(r"(\d+)")[0]
        cat = {}
        for pid, g in vol.groupby("pid"):
            s, e = g["SNFH_CHANGE_OVERALL"].dropna(), g["ET_CHANGE_OVERALL"].dropna()
            cat[pid] = (str(s.iloc[0]) if len(s) else None, str(e.iloc[0]) if len(e) else None)
        presurg = {p["patient"].split("-")[1] for p in P
                   if p["surgery_to_scan1_raw"] is not None and p["surgery_to_scan1_raw"] < 0}
        if presurg != PLAN["alptdg_presurgery"]:
            raise SystemExit(f"[elig] ALPTDG pre-surgery set {sorted(presurg)} != plan {sorted(PLAN['alptdg_presurgery'])}")
        for p in P:
            pid = p["patient"].split("-")[1]
            fa = os.path.join(cfg["cache"], f"{p['patient']}_t1.pt")
            fb = os.path.join(cfg["cache"], f"{p['patient']}_t2.pt")
            fc = os.path.join(cfg["cache"], f"{p['patient']}_change.pt")
            if not (p["usable_basic"] and pid not in nondiff and pid not in presurg
                    and os.path.exists(fa) and os.path.exists(fb)):
                continue
            s2s = p["surgery_to_scan1_raw"]
            diag = str(p["diagnosis"]).strip()
            out.append(base_rec(cohort, p["patient"], 1, 2, fa, fb, p["dt"], reg_dice=p["reg_dice"],
                                diagnosis=diag, gbm=diag in ("Glioblastoma", "Gliosarcoma"),
                                intervening_surgery=bool(p["intervening_surgery"]),
                                pdgm_overlap=bool(p["pdgm_overlap"]),
                                s2s=s2s if (s2s is not None and 0 <= s2s <= 10000) else None,
                                change_file=fc if os.path.exists(fc) else None,
                                change_snfh=cat.get(pid, (None, None))[0], change_et=cat.get(pid, (None, None))[1]))
        if len(out) != PLAN["alptdg_eligible"]:
            raise SystemExit(f"[elig] ALPTDG eligible pairs {len(out)} != plan {PLAN['alptdg_eligible']}")
        log = dict(pairs_in_index=len(P), excluded_nondiffuse=sorted(nondiff), excluded_presurgery=sorted(presurg))
    log.update(eligible_pairs=len(out), eligible_patients=len({r["patient"] for r in out}))
    return out, log


def make_assignment():
    if os.path.exists(ASSIGN):
        raise SystemExit(f"[assign] {ASSIGN} already exists — the assignment is fixed and is never regenerated.")
    os.makedirs(OUT, exist_ok=True)
    res = dict(seed=SEED, K=K, created=datetime.datetime.now().isoformat(timespec="seconds"),
               plan="EXTERNAL_VALIDATION_PLAN v0.4", cohorts={}, eligibility={})
    for c in ("alptdg", "mu"):
        recs, log = eligible_pairs(c)
        pats = sorted({r["patient"] for r in recs})
        perm = np.random.RandomState(SEED).permutation(len(pats))
        res["cohorts"][c] = {pats[i]: int(j % K) for j, i in enumerate(perm)}
        res["eligibility"][c] = log
        counts = np.bincount(list(res["cohorts"][c].values()), minlength=K).tolist()
        print(f"[assign] {c}: {log['eligible_patients']} patients, {log['eligible_pairs']} pairs | per fold model: {counts}")
    with open(ASSIGN, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[assign] saved {ASSIGN}\n[assign] SHA-256: {EU.sha256(ASSIGN)}  -> record in plan Appendix A")


def smoke_records(n_per_fold):
    ds = CachedPairDataset(EU.CACHE)
    folds = list(patient_kfold(ds, k=K))
    recs = []
    for k in range(K):
        for i in folds[k][1][:n_per_fold]:
            fa, fb = ds.pairs[i]
            wa = int(os.path.basename(fa).split("_week")[1][:-3]); wb = int(os.path.basename(fb).split("_week")[1][:-3])
            recs.append(base_rec("lumiere_smoke", ds.patient_of(i), wa, wb, fa, fb, float((wb - wa) * 7),
                                 fold=k, reg_dice=1.0 - 0.01 * len(recs), gbm=bool(k % 2), change_file="SYNTH",
                                 intervening_surgery=(len(recs) % 7 == 0), pdgm_overlap=(len(recs) % 5 == 0),
                                 s2s=float(100 + 10 * len(recs))))
    return recs


# ───────────────────────────── frozen components ─────────────────────────────
class Frozen:
    def __init__(self, args):
        self.T = {int(k): float(v) for k, v in json.load(open(TFILE)).items()}
        h = EU.sha256(BASELINES)
        if h != EXPECTED["baselines"]:
            raise SystemExit(f"[frozen] FROZEN_BASELINES.pt hash mismatch: {h[:12]}")
        self.B = torch.load(BASELINES, map_location="cpu", weights_only=False)
        self.ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=args.R, chunk=args.chunk, steps=20, ddim_steps=50)
        self._nets, self._base = {}, {}
        self.info = dict(T=self.T, baselines_sha256=h, tfile_sha256=EU.sha256(TFILE), models={})
        self.vq = self.seg = None
        if not args.no_image:
            vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
            ck = torch.load(EU.VQGAN, map_location="cpu", weights_only=False)
            vq.load_state_dict(ck.get("gen", ck), strict=False)
            self.vq = vq.eval().to(DEV)
            hs = EU.sha256(EU.SEG)
            if hs != EXPECTED["seg_head"]:
                raise SystemExit(f"[frozen] segmentation head hash mismatch: {hs[:12]}")
            seg = ImgSegUNet().to(DEV)
            seg.load_state_dict(torch.load(EU.SEG, map_location=DEV, weights_only=False)["net"])
            self.seg = seg.eval()
            for p in list(self.vq.parameters()) + list(self.seg.parameters()):
                p.requires_grad_(False)
            self.info["seg_head_sha256"] = hs

    def net(self, k):
        if k not in self._nets:
            net, _, info = EU.load_model(self.ns, k)                  # hash-verified against the manifest
            self._nets[k] = net
            self.info["models"][f"fold{k}"] = info
        return self._nets[k]

    def base(self, k, mode):
        key = (k, mode)
        if key not in self._base:
            b = self.B[k][mode]
            self._base[key] = {n: b[n].to(DEV).float() for n in ("alpha", "beta", "sigma")}
        return self._base[key]


# ───────────────────────────── metrics ─────────────────────────────
def dist_at(S, y, Ts):
    """S [R,n], y [n] -> {T: (fair CRPS, coverage at NOM)} for widening factors Ts."""
    R = S.shape[0]
    mean = S.mean(0)
    Ss, _ = S.sort(0)
    w = 2 * torch.arange(1, R + 1, device=S.device, dtype=S.dtype) - R - 1
    spread = (w[:, None] * Ss).sum(0) / (R * (R - 1))
    qs = torch.tensor(np.concatenate([(1 - NOM) / 2, 1 - (1 - NOM) / 2]), device=S.device, dtype=S.dtype)
    Q = torch.quantile(S, qs, dim=0)
    lo, hi = Q[:len(NOM)], Q[len(NOM):]
    out = {}
    for T in Ts:
        Xt = mean + T * (S - mean)
        crps = ((Xt - y).abs().mean(0) - T * spread).mean().item()
        cov = ((y >= mean + T * (lo - mean)) & (y <= mean + T * (hi - mean))).float().mean(1).cpu().numpy()
        out[T] = (crps, cov)
    return out


def auc(scores, labels):
    s = scores.detach().cpu().numpy().astype(np.float64)
    lab = labels.detach().cpu().numpy().astype(bool)
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_seed(rec, seed):
    return seed * 100003 + zlib.crc32(f"{rec['cohort']}|{rec['patient']}|{rec['tp_a']}|{rec['tp_b']}".encode()) % 1000003


@torch.no_grad()
def eval_pair(rec, fz, args):
    a = torch.load(rec["file_a"], weights_only=False)
    b = torch.load(rec["file_b"], weights_only=False)
    z_a = a["z"].float()[None].to(DEV); z_b = b["z"].float()[None].to(DEV)
    C = z_b.shape[1]
    seg_a = a["seg"].to(DEV).long().reshape(*a["seg"].shape[-3:])
    seg_b = b["seg"].to(DEV).long().reshape(*b["seg"].shape[-3:])
    brain = b["mask"].to(DEV).bool().reshape(*b["mask"].shape[-3:])
    brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
    roi_tok = EU.roi_latent(seg_a, seg_b, EU.CORE)
    regions = {"whole": None, "brain": brain_tok, "roi": roi_tok,
               "roiwt": EU.roi_latent(seg_a, seg_b, EU.WHOLE_TUMOR)}
    k = int(rec["fold"]); T = fz.T[k]
    dt_used = float(rec["dt"]) if rec["dt"] is not None else DT_FALLBACK
    seed = pair_seed(rec, args.seed)

    torch.manual_seed(seed)
    samp = EU.sample_ensemble(fz.ns, fz.net(k), None, z_a, torch.tensor([dt_used], device=DEV))
    mean = samp.mean(0)
    preds, gauss = {"flow": mean, "copy": z_a}, {}
    for mode in ("ar1", "damped"):
        bb = fz.base(k, mode)
        mu = (bb["alpha"] + bb["beta"] * z_a[0])[None]
        preds[mode], gauss[mode] = mu, (mu, bb["sigma"])
    if rec.get("file_prev"):
        pz = torch.load(rec["file_prev"], weights_only=False)
        da, dp, db = a.get("day"), pz.get("day"), b.get("day")
        if None not in (da, dp, db) and da - dp > 0:
            ratio = min((db - da) / (da - dp), 1.0)
            preds["linear"] = z_a + ratio * (z_a - pz["z"].float()[None].to(DEV))

    out = {kk: v for kk, v in rec.items() if not kk.startswith("file_") and kk != "change_file"}
    out.update(T=T, dt_used=dt_used, roi_tokens=int(roi_tok.sum()), has_linear="linear" in preds)
    for m, pred in preds.items():
        for reg, tm in regions.items():
            out[f"mse_{reg}_{m}"], out[f"cos_{reg}_{m}"] = EU.masked_mse_cos(pred, z_b, tm)

    S_all, y_all = samp[:, 0].reshape(args.R, -1), z_b[0].reshape(-1)
    for reg in ("brain", "roi"):
        tm = regions[reg]
        em = tm[None].expand(C, *tm.shape).reshape(-1)
        if int(em.sum()) == 0:
            for key in ("crps_raw_%s_flow", "crps_cal_%s_flow", "crps_%s_ar1", "crps_%s_damped"):
                out[key % reg] = float("nan")
            continue
        d = dist_at(S_all[:, em], y_all[em], [1.0, T])
        out[f"crps_raw_{reg}_flow"], cov_raw = d[1.0]
        out[f"crps_cal_{reg}_flow"], cov_cal = d[T]
        if reg == "brain":
            out["cov_raw_brain_flow"], out["cov_cal_brain_flow"] = cov_raw.tolist(), cov_cal.tolist()
        for mode, (mu, sig) in gauss.items():
            g = torch.Generator(device=DEV).manual_seed(seed + 1)
            Sg = (mu[0] + sig * torch.randn(args.R, *mu.shape[1:], generator=g, device=DEV)).reshape(args.R, -1)[:, em]
            crps_g, cov_g = dist_at(Sg, y_all[em], [1.0])[1.0]
            out[f"crps_{reg}_{mode}"] = crps_g
            if reg == "brain":
                out[f"cov_brain_{mode}"] = cov_g.tolist()

    err_tok = (mean - z_b).pow(2).mean(1)[0].sqrt()
    unc_tok = T * samp.var(0).mean(1)[0].sqrt()
    for reg in ("brain", "roi"):
        tm = regions[reg]
        out[f"errunc_{reg}"] = EU.spearman(err_tok[tm], unc_tok[tm]) if int(tm.sum()) >= 10 else float("nan")
    out["roi_unc_mean"] = float(unc_tok[roi_tok].mean()) if int(roi_tok.sum()) else float("nan")

    if rec.get("change_file"):
        if rec["change_file"] == "SYNTH":                        # smoke test only
            changed = (seg_b > 0) ^ (seg_a > 0)
        else:
            ch = torch.load(rec["change_file"], weights_only=False)
            parts = [ch[kk].to(DEV) > 0 for kk in ("flair", "t1ce") if kk in ch]
            changed = parts[0] | parts[1] if len(parts) == 2 else (parts[0] if parts else torch.zeros_like(brain))
        ctok = F.avg_pool3d(changed.float().reshape(1, 1, *changed.shape[-3:]), 4)[0, 0] > 0.10
        out["change_auc"] = auc(unc_tok[brain_tok], ctok[brain_tok])
        out["changed_tokens"] = int((ctok & brain_tok).sum())

    if fz.vq is not None:
        x_b, x_a = fz.vq.decode(z_b)[0], fz.vq.decode(z_a)[0]
        x_im = torch.stack([fz.vq.decode(samp[r])[0] for r in range(min(K_DECODE, args.R))]).mean(0)
        for name, X in (("forecast", x_im), ("copy", x_a)):
            out[f"psnr_{name}"] = EU.psnr_masked(X, x_b, brain)
            out[f"ssim_{name}"] = EU.ssim_masked(X, x_b, brain)
        labs = {"forecast": fz.seg(x_im[None]).argmax(1)[0], "pipeline_copy": fz.seg(x_a[None]).argmax(1)[0],
                "ceiling": fz.seg(x_b[None]).argmax(1)[0]}
        for src, lab in labs.items():
            for reg, cls in SEG_EVAL.items():
                out[f"dice_{src}_{reg}"] = EU.dice(EU.label_mask(lab, cls), EU.label_mask(seg_b, cls))
    return out


def run(cohort, recs, args):
    d = os.path.join(OUT, cohort); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "per_pair.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done.add((r["patient"], r["tp_a"], r["tp_b"]))
    fz = Frozen(args)
    todo = [r for r in recs if (r["patient"], r["tp_a"], r["tp_b"]) not in done]
    print(f"[run] {cohort}: {len(recs)} pairs, {len(done)} already done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(todo):
            f.write(json.dumps(eval_pair(r, fz, args)) + "\n"); f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                print(f"    {cohort}: {i + 1}/{len(todo)} | {el / (i + 1):.1f}s/pair | "
                      f"ETA {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
                torch.cuda.empty_cache()
    meta = dict(cohort=cohort, finished=datetime.datetime.now().isoformat(timespec="seconds"),
                plan_commit=args.plan_commit, R=args.R, steps=20, seed=args.seed, image_metrics=not args.no_image,
                assignment_sha256=EU.sha256(ASSIGN) if os.path.exists(ASSIGN) else None, **fz.info)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"[run] {cohort} done in {(time.time() - t0) / 60:.1f} min -> {path}")


# ───────────────────────────── statistics ─────────────────────────────
def load_recs(cohort):
    p = os.path.join(OUT, cohort, "per_pair.jsonl")
    if not os.path.exists(p):
        return []
    recs = [json.loads(l) for l in open(p)]
    for r in recs:
        r.setdefault("fold", 0)
    return recs


def safe_test(a, b):
    try:
        return AG.paired_test(a, b)
    except Exception as e:
        return dict(error=str(e), p=float("nan"), n_patients=int(np.isfinite(np.asarray(a, float)).sum()))


def pair(recs, ka, kb):
    return safe_test(*AG.aligned_within(recs, ka, kb))


def calib_test(recs, ideal, key_a="cov_raw_brain_flow", key_b="cov_cal_brain_flow", n_boot=5000):
    byp = {}
    for r in recs:
        if key_a in r and key_b in r:
            byp.setdefault(r["patient"], [[], []])
            byp[r["patient"]][0].append(r[key_a]); byp[r["patient"]][1].append(r[key_b])
    pats = sorted(byp)
    if len(pats) < 3:
        return dict(error="too few patients", p=float("nan"), n_patients=len(pats))
    A = np.stack([np.nanmean(np.array(byp[p][0]), 0) for p in pats])
    B = np.stack([np.nanmean(np.array(byp[p][1]), 0) for p in pats])
    ce = lambda M: float(np.mean(np.abs(np.nanmean(M, 0) - ideal)))
    obs = ce(A) - ce(B)
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(pats), (n_boot, len(pats)))
    d = np.array([ce(A[i]) - ce(B[i]) for i in idx])
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / n_boot)
    return dict(n_patients=len(pats), calerr_raw=ce(A), calerr_cal=ce(B), improvement=obs,
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p=float(min(p, 1.0)))


def primary(recs, ideal):
    return {"H1a ROI MSE flow vs AR(1)": pair(recs, "mse_roi_flow", "mse_roi_ar1"),
            "H1b ROI calibrated CRPS flow vs AR(1)": pair(recs, "crps_cal_roi_flow", "crps_roi_ar1"),
            "H2 calibration error T_k vs T=1": calib_test(recs, ideal)}


def summarize(cohorts, args):
    ideal = AG.ideal_coverage(args.R, NOM)
    res, prim, sec = {"ideal_coverage": ideal.tolist(), "cohorts": {}}, [], []
    for c in cohorts:
        R = load_recs(c)
        if not R:
            print(f"[sum] {c}: no records"); continue
        S = dict(n_pairs=len(R), n_patients=len({r["patient"] for r in R}))
        S["primary"] = primary(R, ideal)
        prim += [(c, k_, v) for k_, v in S["primary"].items()]
        S["secondary"] = {
            "brain MSE flow vs AR(1)": pair(R, "mse_brain_flow", "mse_brain_ar1"),
            "whole MSE flow vs AR(1)": pair(R, "mse_whole_flow", "mse_whole_ar1"),
            "ROI cosine flow vs AR(1) (higher is better)": pair(R, "cos_roi_flow", "cos_roi_ar1"),
            "ROI raw CRPS flow vs AR(1)": pair(R, "crps_raw_roi_flow", "crps_roi_ar1"),
            "ROI MSE flow vs damped": pair(R, "mse_roi_flow", "mse_roi_damped"),
            "ROI calibrated CRPS flow vs damped": pair(R, "crps_cal_roi_flow", "crps_roi_damped"),
            "ROI MSE flow vs Copy": pair(R, "mse_roi_flow", "mse_roi_copy"),
        }
        sec += [(c, k_, v) for k_, v in S["secondary"].items()]
        pm = lambda key: np.array([v for _, v in AG.by_patient(R, key).values()], float)
        S["descriptive"] = {
            "calibration_error_AR1": calib_test(R, ideal, "cov_brain_ar1", "cov_brain_ar1").get("calerr_raw"),
            "errunc_brain_mean": float(np.nanmean(pm("errunc_brain"))),
            "errunc_roi_mean": float(np.nanmean(pm("errunc_roi"))),
        }
        u, e = pm("roi_unc_mean"), pm("mse_roi_flow")
        ok = np.isfinite(u) & np.isfinite(e)
        if ok.sum() >= 5:
            rho, pv = spearmanr(u[ok], e[ok])
            S["descriptive"]["case_level_uncertainty_vs_error"] = dict(spearman=float(rho), p=float(pv), n=int(ok.sum()))
        if any("change_auc" in r for r in R):
            a_ = pm("change_auc"); a_ = a_[np.isfinite(a_)]
            boot = [np.mean(a_[np.random.RandomState(i).randint(0, len(a_), len(a_))]) for i in range(2000)] if len(a_) else [np.nan]
            S["descriptive"]["change_auc"] = dict(mean=float(np.mean(a_)) if len(a_) else None, n=int(len(a_)),
                                                  ci95=[float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))])
        for key in ("psnr_forecast", "psnr_copy", "ssim_forecast", "ssim_copy"):
            if any(key in r for r in R):
                S["descriptive"][key] = float(np.nanmean(pm(key)))
        for src in ("forecast", "pipeline_copy", "ceiling"):
            for reg in SEG_EVAL:
                key = f"dice_{src}_{reg}"
                if any(key in r for r in R):
                    S["descriptive"][key] = float(np.nanmean(pm(key)))

        sens = {}
        dice_by_p = {p: v for p, (_, v) in AG.by_patient(R, "reg_dice").items()}
        order = sorted(dice_by_p, key=lambda p: (dice_by_p[p], p))
        drop = set(order[:int(np.floor(0.1 * len(order)))])
        sens["registration: lowest decile excluded"] = dict(excluded=len(drop), **primary([r for r in R if r["patient"] not in drop], ideal))
        if c in ("alptdg", "lumiere_smoke"):
            sens["intervening surgery excluded"] = primary([r for r in R if not r.get("intervening_surgery")], ideal)
            sens["UCSF-PDGM overlap excluded"] = primary([r for r in R if not r.get("pdgm_overlap")], ideal)
        S["sensitivity"] = sens

        sub = {}
        def grp(name, sel):
            rr = [r for r in R if sel(r)]
            if len({r["patient"] for r in rr}) >= 3:
                sub[name] = dict(roi_mse=pair(rr, "mse_roi_flow", "mse_roi_ar1"),
                                 roi_crps=pair(rr, "crps_cal_roi_flow", "crps_roi_ar1"))
        grp("histology: glioblastoma", lambda r: r.get("gbm"))
        grp("histology: other diffuse glioma", lambda r: not r.get("gbm"))
        grp("dt within LUMIERE 5-95% range", lambda r: r.get("dt") is not None and DT_RANGE[0] <= r["dt"] <= DT_RANGE[1])
        grp("dt outside LUMIERE range", lambda r: r.get("dt") is not None and not (DT_RANGE[0] <= r["dt"] <= DT_RANGE[1]))
        s2 = [r["s2s"] for r in R if r.get("s2s") is not None]
        if s2:
            med = float(np.median(s2))
            grp(f"time since surgery <= median ({med:.0f} d)", lambda r: r.get("s2s") is not None and r["s2s"] <= med)
            grp(f"time since surgery > median ({med:.0f} d)", lambda r: r.get("s2s") is not None and r["s2s"] > med)
        for v in sorted({str(r.get("change_snfh")) for r in R if r.get("change_snfh") is not None}):
            grp(f"SNFH change category: {v}", lambda r, v=v: str(r.get("change_snfh")) == v)
        S["subgroups"] = sub
        res["cohorts"][c] = S

    for fam, name in ((prim, "primary"), (sec, "secondary")):
        ps = [v.get("p", np.nan) for _, _, v in fam]
        ok = [i for i, p in enumerate(ps) if np.isfinite(p)]
        adj = AG.holm([ps[i] for i in ok]) if ok else []
        for i, a_ in zip(ok, adj):
            fam[i][2]["p_holm"] = float(a_)
        res[f"holm_family_{name}"] = f"{len(ok)} tests across cohorts {cohorts}"

    tag = "_".join(cohorts)
    os.makedirs(os.path.join(OUT, "summary"), exist_ok=True)
    out = os.path.join(OUT, "summary", f"results_{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)

    print("=" * 96)
    print(f"  EXTERNAL VALIDATION — primary endpoints (Holm across {len([1 for *_, v in prim if 'p_holm' in v])} tests)")
    print("=" * 96)
    for c, name, v in prim:
        if "calerr_raw" in v:
            print(f"  {c:8s} {name:40s} raw {v['calerr_raw']:.4f} -> cal {v['calerr_cal']:.4f}  "
                  f"Δ {v['improvement']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  p_holm={v.get('p_holm', np.nan):.1e}  n={v['n_patients']}")
        elif "mean_a" in v:
            print(f"  {c:8s} {name:40s} {v['mean_a']:.4f} vs {v['mean_b']:.4f}  rel {v['rel_improvement_pct']:+5.1f}% "
                  f"[{v['ci95_pct'][0]:+.1f},{v['ci95_pct'][1]:+.1f}]  p_holm={v.get('p_holm', np.nan):.1e}  n={v['n_patients']}")
        else:
            print(f"  {c:8s} {name:40s} {v}")
    print("-" * 96)
    for c in res["cohorts"]:
        print(f"  {c}: secondary, sensitivity, subgroups and descriptive results in the JSON")
    print(f"[saved] {out}")


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_pairs", type=int, default=2, help="LUMIERE validation pairs per fold in smoke mode")
    ap.add_argument("--cohort", choices=["mu", "alptdg"])
    ap.add_argument("--plan_commit", type=str, default="")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--cohorts", nargs="+", default=["alptdg", "mu"])
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_image", action="store_true")
    args = ap.parse_args()

    if args.assign:
        make_assignment()
    elif args.smoke:
        run("lumiere_smoke", smoke_records(args.smoke_pairs), args)
        summarize(["lumiere_smoke"], args)
    elif args.cohort:
        if len(args.plan_commit) < 7:
            raise SystemExit("[rule0] Refusing to run on external data without --plan_commit <hash of the frozen plan>.")
        if not os.path.exists(ASSIGN):
            raise SystemExit("[rule0] No assignment file. Run --assign and commit it with the plan first.")
        A = json.load(open(ASSIGN))
        recs, log = eligible_pairs(args.cohort)
        assigned = A["cohorts"][args.cohort]
        if {r["patient"] for r in recs} != set(assigned):
            raise SystemExit("[rule0] Eligible patients differ from the committed assignment — stop and investigate.")
        for r in recs:
            r["fold"] = assigned[r["patient"]]
        print(f"[run] eligibility: {log}")
        run(args.cohort, recs, args)
    elif args.summarize:
        summarize(args.cohorts, args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
