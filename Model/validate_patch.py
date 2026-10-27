# ─── DROP-IN REPLACEMENT for validate() in train.py ───
# Fixes the 16 GiB OOM: decode ONE volume at a time (not the whole batch),
# and free memory aggressively. Paste this over the existing validate().

@torch.no_grad()
def validate(net, ema, loader, vqgan, cfg, device):
    backup = copy.deepcopy(net.state_dict()); ema.copy_to(net); net.eval()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)

    agg = {}
    n_img = 0
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

        # image metrics: decode ONE volume at a time (avoids the 16 GiB OOM)
        if cfg["val_image_metrics"] and n_img < cfg["val_image_n"]:
            for i in range(z_b.shape[0]):
                if n_img >= cfg["val_image_n"]:
                    break
                xp = vqgan.decode(mean[i:i+1])          # [1,C,128,128,128]
                xt = vqgan.decode(z_b[i:i+1])
                mk = b["mask_b"][i:i+1].to(device)
                for k, v in M.image_fidelity(xp, xt, mk).items():
                    agg.setdefault(k, []).append(v)
                del xp, xt
                torch.cuda.empty_cache()
                n_img += 1
        del samples, mean
        torch.cuda.empty_cache()
    net.load_state_dict(backup)
    return {k: float(sum(v) / len(v)) for k, v in agg.items()}
