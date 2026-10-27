#!/usr/bin/env python3
"""
apply_bio_patch.py — add the soft biophysics regularizer to train.py (option ii).

Adds to the CONFIG:  bio_lambda, bio_every, bio_residual_scale, use_param_head, seg_ckpt
Wires into training: every `bio_every` steps, compute a single-step forecast (few ODE
steps), run BioRegularizer, add bio_lambda * L_bio to the loss. Logs L_bio, D_med, rho_med.

Idempotent; backs up train.py -> train.py.bak3. Run once in the Bio-SSM-Flow folder.
"""
import os, re, shutil
P = "train.py"; assert os.path.exists(P)
shutil.copy(P, P + ".bak3"); src = open(P).read()

# 1) config additions
if "bio_lambda" not in src:
    src = src.replace(
        '    RESUME = None,',
        '    RESUME = None,\n'
        '    bio_lambda = 0.05,          # SOFT physics weight (gentle; ~20% prior)\n'
        '    bio_every = 10,             # apply biophysics every N steps (cost control)\n'
        '    bio_residual_scale = 3.6e-5,# calibrated from real |dc/dt|\n'
        '    use_param_head = True,      # learn patient-specific (D,rho)\n'
        '    seg_ckpt = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt",',
        1)

# 2) imports + build regularizer after vqgan is built
if "from bio_coupling import BioRegularizer" not in src:
    src = src.replace("import metrics as M", "import metrics as M\nfrom bio_coupling import BioRegularizer\nfrom seg_head_img import ImgSegUNet\nfrom flow import generate as _bio_generate", 1)

# build the regularizer right after the vqgan freeze loop
anchor = "    for p in vqgan.parameters(): p.requires_grad_(False)"
if anchor in src and "bio_reg = " not in src:
    build = (anchor + "\n\n"
        "    # ---- soft biophysics regularizer (option ii) ----\n"
        "    bio_reg = None\n"
        "    if cfg.get('bio_lambda', 0) > 0 and os.path.exists(cfg.get('seg_ckpt','')):\n"
        "        _seg = ImgSegUNet().to(device)\n"
        "        _seg.load_state_dict(torch.load(cfg['seg_ckpt'], map_location=device, weights_only=False)['net'])\n"
        "        _seg.eval()\n"
        "        for p in _seg.parameters(): p.requires_grad_(False)\n"
        "        bio_reg = BioRegularizer(vqgan, _seg, device,\n"
        "                                 residual_scale=cfg['bio_residual_scale'],\n"
        "                                 use_param_head=cfg['use_param_head'])\n"
        "        if bio_reg.parameters():\n"
        "            opt.add_param_group({'params': bio_reg.parameters(), 'lr': cfg['lr']})\n"
        "        print(f\"[train] biophysics ON: lambda={cfg['bio_lambda']} every={cfg['bio_every']}\")\n")
    src = src.replace(anchor, build, 1)

# 3) inside the training step, after the FM loss is computed, add periodic bio term.
#    We hook right before 'opt.zero_grad' in the main loop.
step_anchor = "            opt.zero_grad(set_to_none=True); loss.backward()"
if step_anchor in src and "bio_reg is not None and step %" not in src:
    inject = (
        "            if bio_reg is not None and step % cfg['bio_every'] == 0 and step >= cfg['var_warmup']:\n"
        "                with torch.no_grad():\n"
        "                    zpred = _bio_generate(lambda z,ta,za,d: net(z,ta,za,d), z_a, dt, steps=6, n_samples=1)[0]\n"
        "                zpred = zpred.detach().requires_grad_(True)\n"
        "                # re-generate WITH grad through a light 2-step path for coupling\n"
        "                eps = torch.randn_like(z_b); zc = eps\n"
        "                for _m in range(6):\n"
        "                    ta = torch.full((z_a.shape[0],), _m/6, device=device)\n"
        "                    vv,_ = net(zc, ta, z_a, dt); zc = zc + (1/6)*((net(zc,ta,z_a,dt)[0]-zc)/max(1-_m/6,1e-2))\n"
        "                Lbio, biolog = bio_reg(z_a, zc, b['seg_a'].to(device), dt, b['mask_b'].to(device))\n"
        "                loss = loss + cfg['bio_lambda'] * Lbio\n"
        "                if step % 100 == 0: print(f\"    [bio step {step}] L_bio={biolog.get('L_bio',0):.4f} D={biolog.get('D_med',0):.4f} rho={biolog.get('rho_med',0):.4f}\")\n"
        + step_anchor)
    src = src.replace(step_anchor, inject, 1)
    print("bio term injected into training loop.")
else:
    print("WARNING: could not find loss.backward anchor; check train.py loop.")

open(P,"w").write(src)
print("Patched train.py (backup train.py.bak3). Biophysics = option (ii), soft + periodic.")
print("NOTE: cached pairs must include seg_a; CachedPairDataset already returns it.")
