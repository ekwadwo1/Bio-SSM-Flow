#!/usr/bin/env python3
"""
patch_ddpm_warmup.py — make the DDPM transport use the SAME variance-warmup as the
flow recipe (pure eps-MSE for the first var_warmup steps, then heteroscedastic NLL),
so the flow-vs-diffusion comparison differs ONLY in the transport. Run once in the
Bio-SSM-Flow folder. Backs up to train_ddpm.py.bak.
"""
import os, shutil
P = "train_ddpm.py"
assert os.path.exists(P), "run in Bio-SSM-Flow (train_ddpm.py must be here)"
shutil.copy(P, P + ".bak")
src = open(P).read()

# 1) DDPMTransport.training_loss returns (loss, logs) via the heteroscedastic loss_fn.
#    We add a var_warmup path in train_ddpm's loop: before var_warmup, recompute a
#    pure eps-MSE loss instead. Easiest: add a 'warmup' flag to training_loss.
old_call = "            with torch.amp.autocast(dev,dtype=torch.bfloat16):\n                loss,logs=ddpm.training_loss(z_b,z_a,dt,loss_fn)"
new_call = ("            with torch.amp.autocast(dev,dtype=torch.bfloat16):\n"
            "                loss,logs=ddpm.training_loss(z_b,z_a,dt,loss_fn,\n"
            "                                             warmup=(step<cfg['var_warmup']))")
assert old_call in src, "training_loss call not found (already patched?)"
src = src.replace(old_call, new_call, 1)
open(P,"w").write(src)
print("patched train_ddpm.py call site (added warmup flag)")

# 2) Patch DDPMTransport.training_loss in ddpm_transport.py to honor warmup.
P2 = "ddpm_transport.py"
shutil.copy(P2, P2 + ".bak")
s2 = open(P2).read()
old_fn = ('''    def training_loss(self, z_b, z_a, dt, loss_fn):
        B = z_b.shape[0]; device = z_b.device
        s_idx = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(z_b)
        z_s = self.q_sample(z_b, s_idx, eps)
        # net expects a normalized time in the sinusoidal range; pass s_idx directly
        # (VelocityNetV2 rescales tau*T_EMB; here we pass s_idx/T so *T_EMB lands in range)
        s_norm = (s_idx.float() / self.T)
        eps_hat, logvar = self.net(z_s, s_norm, z_a, dt)
        # heteroscedastic NLL on eps (same structure as the flow HeteroscedasticFMLoss)
        loss, v_tilde, logs = loss_fn(eps, eps_hat, logvar)
        return loss, logs''')
new_fn = ('''    def training_loss(self, z_b, z_a, dt, loss_fn, warmup=False):
        B = z_b.shape[0]; device = z_b.device
        s_idx = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(z_b)
        z_s = self.q_sample(z_b, s_idx, eps)
        s_norm = (s_idx.float() / self.T)
        eps_hat, logvar = self.net(z_s, s_norm, z_a, dt)
        if warmup:
            # variance-warmup: pure eps-MSE, MATCHING the flow recipe's first phase
            loss = torch.nn.functional.mse_loss(eps_hat.float(), eps.float())
            logs = {"vel_mse": float(loss.detach()), "sigma2_mean": float("nan")}
        else:
            loss, v_tilde, logs = loss_fn(eps, eps_hat, logvar)
        return loss, logs''')
assert old_fn in s2, "training_loss body not found (already patched?)"
s2 = s2.replace(old_fn, new_fn, 1)
open(P2,"w").write(s2)
print("patched ddpm_transport.py training_loss (warmup -> pure eps-MSE)")
print("DONE. DDPM now matches the flow variance-warmup schedule exactly.")
