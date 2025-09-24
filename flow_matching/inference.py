import os
import torch

from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision import transforms

from models import SiT_B_8
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from dataset import SiTDataset

# --------------- Configuration ----------------
device = "cuda:0"
image_size = 128
num_classes = 1
vae_type = "ema"
ckpt_path = "results/007-SiT-B-8-Linear-velocity-None/checkpoints/0040000.pt"
dataset_dir = "data/chair_data/train"
N = 10
output_dir = "pose_eval/final/0040000"
os.makedirs(output_dir, exist_ok=True)

# --------------- Load Dataset -----------------
latent_size = image_size // 8
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
])
dataset = SiTDataset(dataset_dir, transform=transform, image_size=image_size)

# --------------- Load VAE --------------------
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
vae.eval()

# --------------- Load Model ------------------
state_dict = find_model(ckpt_path)
model = SiT_B_8(input_size=latent_size, num_classes=num_classes).to(device)
model.load_state_dict(state_dict)
model.eval()

# --------------- Sampler Setup ----------------
transport = create_transport()
sampler = Sampler(transport)
sample_fn = sampler.sample_ode(
    sampling_method="dopri5",
    atol=1e-6,
    rtol=1e-3,
    num_steps=250
)

# --------------- Inference Loop --------------
with torch.no_grad():
    for i in range(N):
        image, gaussian_pose, gt_pose, _ = dataset[i]

        image = image.unsqueeze(0).to(device)
        gaussian_pose = gaussian_pose.unsqueeze(0).to(device)

        n = 1
        y = torch.IntTensor([0]).to(device)
        z = torch.cat([gaussian_pose, gaussian_pose], 0)
        y_null = torch.tensor([1] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=4)

        samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
        img_pred = vae.decode(samples / 0.18215).sample[0].unsqueeze(0)

        # Resize if needed
        if image.shape[2:] != img_pred.shape[2:]:
            image = F.interpolate(image, size=img_pred.shape[2:], mode='bilinear', align_corners=False)

        comparison = torch.cat([img_pred.cpu(), image.cpu()], dim=0)
        save_image(comparison, os.path.join(output_dir, f"compare_{i}.png"),
                   nrow=2, normalize=True, value_range=(-1, 1))

        print(f"Saved {output_dir}/compare_{i}.png: [0]=Generated, [1]=Original")

print("All comparisons saved in", output_dir)
