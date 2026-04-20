import torch

ckpt = torch.load("experiments/euroc/motion_body_rot/ckpt/best_model.ckpt", map_location="cpu")

print(type(ckpt))

if isinstance(ckpt, dict):
    print(ckpt.keys())