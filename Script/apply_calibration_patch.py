#!/usr/bin/env python3
"""
apply_calibration_patch.py — bake per-fold temperature calibration into train.py.

Changes (idempotent; backs up to train.py.bak2):
  (1) import calibration
  (2) in validate(): collect per-pair ensembles, FIT T on val, apply T, and report
      BOTH raw and calibrated PICP + the fitted T. T is saved into the returned dict.
  (3) best.pt now also stores the fitted 'temperature' for inference.

Run once in the Bio-SSM-Flow folder:  python apply_calibration_patch.py
"""
import os, re, shutil
P = "train.py"
assert os.path.exists(P), "run in Bio-SSM-Flow (train.py must be here)"
shutil.copy(P, P + ".bak2")
src = open(P).read()

# (1) import
if "import calibration" not in src:
    src = src.replace("import metrics as M", "import metrics as M\nimport calibration as CAL", 1)

# (2) replace validate() with a T-fitting version
NEW = '''@torch.no_grad()
def validate(net, ema, loader, vqgan, cfg, device):
    backup = copy.deepcopy(net.state_dict()); ema.copy_to(net); net.eval()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)
    agg = {}; n_img = 0
    per_samp, per_y = [], []
    for bi, b in enumerate(loader):
        z_a = b["z_a"].to(device); z_b = b["z_b"].to(device); dt = b["dt"].to(device)
        net.set_week(None)
        samples = generate(v_fn, z_a, dt, steps=cfg["gen_steps"], n_samples=cfg["gen_samples"])
        # store per-pair ensembles for temperature fitting (split batch into singletons)
        for j in range(z_b.shape[0]):
            per_samp.append(samples[:, j:j+1]); per_y.append(z_b[j:j+1])
        mean = samples.mean(0)
        for k, v in M.latent_fidelity(mean, z_b).items(): agg.setdefault(k, []).append(v)
        agg.setdefault("crps", []).append(M.ensemble_crps(samples, z_b))
        agg.setdefault("picp90_raw", []).append(M.picp(samples, z_b, 0.9))
        agg.setdefault("diversity", []).append(M.sample_diversity(samples))
        s2 = samples.var(0).mean(1).reshape(z_b.shape[0], -1).clamp_min(1e-8)
        for k, v in M.uncertainty_block(mean, z_b, s2).items(): agg.setdefault(k, []).append(v)
        if cfg["val_image_metrics"] and n_img < cfg["val_image_n"]:
            for i in range(z_b.shape[0]):
                if n_img >= cfg["val_image_n"]: break
                xp = vqgan.decode(mean[i:i+1]); xt = vqgan.decode(z_b[i:i+1])
                mk = b["mask_b"][i:i+1].to(device)
                for k, v in M.image_fidelity(xp, xt, mk).items(): agg.setdefault(k, []).append(v)
                del xp, xt; torch.cuda.empty_cache(); n_img += 1
        del samples, mean; torch.cuda.empty_cache()
    # FIT temperature on validation ensembles, then report calibrated coverage
    T = CAL.fit_temperature(per_samp, per_y, verbose=False)
    picp90_cal = float(sum(CAL._picp(CAL.calibrate(s, T), y, 0.9) for s, y in zip(per_samp, per_y)) / max(len(per_y),1))
    cov_err = CAL.coverage_error(per_samp, per_y, T)
    net.load_state_dict(backup)
    out = {k: float(sum(v) / len(v)) for k, v in agg.items()}
    out["temperature"] = float(T); out["picp90_cal"] = picp90_cal; out["cov_err_cal"] = cov_err
    return out
'''
pat = re.compile(r"@torch\.no_grad\(\)\s*\ndef validate\(.*?\n(?=\n(?:def |@|class ))", re.S)
if pat.search(src):
    src = pat.sub(NEW + "\n", src); print("validate() replaced with T-fitting version.")
else:
    print("WARNING: validate() not found by regex.")

# (3) store temperature in best.pt
src = src.replace(
    'torch.save({"net": net.state_dict(), "ema": ema.shadow, "step": step,\n'
    '                                "best_crps": best_crps, "cfg": cfg},',
    'torch.save({"net": net.state_dict(), "ema": ema.shadow, "step": step,\n'
    '                                "best_crps": best_crps, "temperature": vm.get("temperature",1.0), "cfg": cfg},')

open(P,"w").write(src)
print("Patched train.py (backup train.py.bak2).")
print("val log will now include: temperature, picp90_cal, cov_err_cal, picp90_raw")
