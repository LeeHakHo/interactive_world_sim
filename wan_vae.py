"""Wan2.2 TI2V VAE 封装 —— 官方途径 (diffusers AutoencoderKLWan + 官方权重/config)。

权威源: Wan-AI/Wan2.2-TI2V-5B-Diffusers 的 vae/ 子目录 (config.json + diffusion_pytorch_model.safetensors),
  下到 ckpts_wan22vae/official_diffusers/vae/。实现 = 官方 diffusers.AutoencoderKLWan (HuggingFace+Wan Team)。
  ★必须用独立解释器 .venv_wan/bin/python (装了 diffusers 0.35.2); iws env 的 diffusers 0.29.2 没有 AutoencoderKLWan。

官方规格 (config.json): base_dim=160, z_dim=48, dim_mult=[1,2,4,4], temperal_downsample=[False,True,True],
  scale_factor_spatial=16, scale_factor_temporal=4, attn_scales=[]。含官方 latents_mean/std(48维)。

接口: WanVAE().encode(x)->z_norm ; .decode(z_norm)->x ; .roundtrip(x)->x_hat。
  x: (B,3,T,H,W) in [0,1] (内部 *2-1 到 [-1,1]);
  z_norm: (B,48,t,H/16,W/16) 官方 mean/std 归一后的 latent (~单位尺度, 供 DiT 训练);  t=1+(T-1)//4。
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

_DEFAULT_DIR = "ckpts_wan22vae/official_diffusers"


class WanVAE(nn.Module):
    def __init__(self, model_dir=_DEFAULT_DIR, device="cuda", dtype=torch.float32):
        super().__init__()
        self.vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype)
        self.vae.eval().to(device)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.device, self.dtype = device, dtype
        z = self.vae.config.z_dim
        # 官方 latent 归一化统计 (逐通道), reshape 成 (1,C,1,1,1) 便于广播
        m = torch.tensor(self.vae.config.latents_mean, dtype=dtype).view(1, z, 1, 1, 1)
        s = torch.tensor(self.vae.config.latents_std, dtype=dtype).view(1, z, 1, 1, 1)
        self.register_buffer("lat_mean", m.to(device))
        self.register_buffer("lat_std", s.to(device))
        self.z_dim = z

    @torch.no_grad()
    def encode(self, x):
        """(B,3,T,H,W)[0,1] -> 归一化 latent (B,48,t,H/16,W/16). deterministic(取分布 mode)。"""
        x = (x.to(self.device, self.dtype) * 2 - 1)
        lat = self.vae.encode(x).latent_dist.mode()
        return (lat - self.lat_mean) / self.lat_std

    @torch.no_grad()
    def decode(self, z_norm):
        """归一化 latent -> (B,3,T,H,W)[0,1]。"""
        lat = z_norm.to(self.device, self.dtype) * self.lat_std + self.lat_mean
        x = self.vae.decode(lat).sample
        return ((x + 1) / 2).clamp(0, 1)

    @torch.no_grad()
    def roundtrip(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    v = WanVAE()
    print(f"z_dim={v.z_dim}  spatial={v.vae.config.scale_factor_spatial}x  temporal={v.vae.config.scale_factor_temporal}x")
    for T, H in [(1, 256), (17, 256)]:
        x = torch.rand(1, 3, T, H, H)
        z = v.encode(x)
        xr = v.decode(z)
        print(f"T={T} H={H}: x{tuple(x.shape)} -> z{tuple(z.shape)} -> x_hat{tuple(xr.shape)}  z_mean={z.mean():.3f} z_std={z.std():.3f}")
