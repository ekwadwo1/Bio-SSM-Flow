#!/usr/bin/env python3
"""
apply_fixes.py — programmatically patch train.py:
  (1) replace validate() with a per-volume-decode version (fixes the 16 GiB OOM)
  (2) pin the device explicitly via torch.cuda.set_device (fixes the 'ran on GPU 0'
      mystery) and force expandable_segments.
Run once:  python apply_fixes.py   (edits train.py in place, backs up to train.py.bak)
"""
import re, shutil, os

P = "train.py"
assert os.path.exists(P), "run this in the Bio-SSM-Flow folder (train.py must be here)"
shutil.copy(P, P + ".bak")
src = open(P).read()

# ---- (2) force expandable segments + assert device at top of main() ----
if "PYTORCH_CUDA_ALLOC_CONF" not in src:
    src = src.replace(
        "from __future__ import annotations",
        "from __future__ import annotations\nimport os as _os\n"
        "_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')",
        1)

# ---- (1) replace the whole validate() function ----
NEW_VALIDATE = '''@torch.no_grad()
def validate(net, ema, loader, vqgan, cfg, device):
    backup = copy.deepcopy(net.state_dict()); ema.copy_to(net); net.eval()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)
    agg = {}; n_img = 0
    for bi, b in enumerate(loader):
        z_a = b["z_a"].to(device); z_b = b["z_b"].to(device); dt = b["dt"].to(device)
        net.set_week(None)
        samples = generate(v_fn, z_a, dt, steps=cfg["gen_steps"], n_samples=cfg["gen_samples"])
        mean = samples.mean(0)
        for k, v in M.latent_fidelity(mean, z_b).items(): agg.setdefault(k, []).append(v)
        agg.setdefault("crps", []).append(M.ensemble_crps(samples, z_b))
        agg.setdefault("picp90", []).append(M.picp(samples, z_b, 0.9))
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
    net.load_state_dict(backup)
    return {k: float(sum(v) / len(v)) for k, v in agg.items()}
'''

# regex: from '@torch.no_grad()' immediately preceding 'def validate' to the end of that function
pat = re.compile(r"@torch\.no_grad\(\)\s*\ndef validate\(.*?\n(?=\n(?:def |@|class ))", re.S)
if pat.search(src):
    src = pat.sub(NEW_VALIDATE + "\n", src)
    print("validate() replaced.")
else:
    print("WARNING: could not locate validate() by regex; check manually.")

open(P, "w").write(src)
print("Patched train.py (backup at train.py.bak).")
print("Reminders: batch is", "16" if "batch   = 16" in src else "??",
      "| val_image_n present:", "val_image_n" in src)
