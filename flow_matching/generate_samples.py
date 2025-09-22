import os

import torch
from torchvision.utils import save_image
from torch.amp import autocast

from models import SiT_models
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL

# ------ Config ------
results_dir = "results/000-SiT-B-8-Linear-velocity-None" 
ema_sampling_dir = os.path.join(results_dir, "ema_sampling_checkpoints")
sample_output_dir = os.path.join(results_dir, "ema_samples")
model_name = "SiT-B/8"
image_size = 128 
num_classes = 1
cfg_scale = 1.0
vae_type = "ema" 
sample_batch_size = 16 
device = "cuda:0"

# --- Prepare everything ---
os.makedirs(sample_output_dir, exist_ok=True)
ckpt_files = sorted([f for f in os.listdir(ema_sampling_dir) if f.endswith('.pt')])

# Load model class and VAE only ONCE if possible
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
vae.eval()

# (Transport/sampler)
transport = create_transport(
    "Linear", "velocity", None, None, None  # replace with actual args if needed
)
transport_sampler = Sampler(transport)

for ckpt_file in ckpt_files:
    ckpt_path = os.path.join(ema_sampling_dir, ckpt_file)
    print(f"Generating samples from {ckpt_path}")
    ema_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ema_ckpt["step"]

    # Build EMA model, load weights
    ema_model = SiT_models[model_name](
        input_size=image_size // 8,
        num_classes=num_classes,
    ).to(device)
    ema_model.load_state_dict(ema_ckpt["ema"])
    ema_model.eval()

    # Setup classifier-free guidance
    n = sample_batch_size
    latent_size = image_size // 8
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    ys = torch.zeros(n, dtype=torch.long, device=device)
    if cfg_scale > 1.0:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=cfg_scale)
        model_fn = ema_model.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema_model.forward

    sample_fn = transport_sampler.sample_ode()
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]

    if cfg_scale > 1.0:
        samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, os.path.join(sample_output_dir, f"sample_{step}.png"),
               nrow=int(4), normalize=True, value_range=(-1, 1))

    print(f"Saved EMA samples for step {step}")

    del ema_model, sample_fn, samples
    torch.cuda.empty_cache()
    import gc; gc.collect()