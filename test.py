import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from core.model.build import build_model
from core.utils.config import load_config


def fuse_mri_pet_color(mri_tensor, pet_tensor, alpha=0.5):
    import cv2
    import numpy as np

    mri = mri_tensor.squeeze().cpu().numpy()
    pet = pet_tensor.squeeze().cpu().numpy()

    mri_img = (mri * 255).astype(np.uint8)
    pet_img = (pet * 255).astype(np.uint8)

    mri_bgr = cv2.cvtColor(mri_img, cv2.COLOR_GRAY2BGR)
    pet_color = cv2.applyColorMap(pet_img, cv2.COLORMAP_HOT)
    fused = cv2.addWeighted(pet_color, alpha, mri_bgr, 1 - alpha, 0)
    return cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)


def run_batch_fusion():
    vis_dir = 'D:/Projects/Pytorch_DDcGAN-master/demo/vis'
    inf_dir = 'D:/Projects/Pytorch_DDcGAN-master/demo/inf'
    vis_mask_dir = 'D:/Projects/Pytorch_DDcGAN-master/demo/vis_mask'
    inf_mask_dir = 'D:/Projects/Pytorch_DDcGAN-master/demo/inf_mask'
    out_dir = 'D:/Projects/Pytorch_DDcGAN-master/demo/fused'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'D:/Projects/Pytorch_DDcGAN-master/config/GAN_G1_D2.yaml'
    weight_path = 'D:/Projects/Pytorch_DDcGAN-master/weights/GAN_G1_D2_0301/Generator/Generator_best.pth'

    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gan_model = build_model(config)
    gan_model.Generator.load_state_dict(torch.load(weight_path, map_location=device).state_dict())
    gan_model = gan_model.to(device)
    gan_model.eval()

    mean = config['Dataset']['mean']
    std = config['Dataset']['std']

    trans_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    trans_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    file_list = sorted(os.listdir(vis_dir))
    for fname in file_list:
        vis_path = os.path.join(vis_dir, fname)
        inf_path = os.path.join(inf_dir, fname)
        vis_mask_path = os.path.join(vis_mask_dir, fname.replace('.png', '_mask.png'))
        inf_mask_path = os.path.join(inf_mask_dir, fname.replace('.png', '_mask.png'))

        if not os.path.exists(vis_path) or not os.path.exists(inf_path):
            print(f"[skip] missing image: {fname}")
            continue
        if not os.path.exists(vis_mask_path) or not os.path.exists(inf_mask_path):
            print(f"[skip] missing mask: {fname}")
            continue

        vis_img = Image.open(vis_path).convert('RGB')
        inf_img = Image.open(inf_path).convert('RGB')
        vis_mask = Image.open(vis_mask_path).convert('L')
        inf_mask = Image.open(inf_mask_path).convert('L')

        vis_tensor = trans_img(vis_img).unsqueeze(0).to(device)
        inf_tensor = trans_img(inf_img).unsqueeze(0).to(device)
        vis_mask_tensor = (trans_mask(vis_mask) > 0.5).float().unsqueeze(0).to(device)
        inf_mask_tensor = (trans_mask(inf_mask) > 0.5).float().unsqueeze(0).to(device)

        generator_data = {
            'Vis': torch.cat([vis_tensor, vis_mask_tensor], dim=1),
            'Inf': torch.cat([inf_tensor, inf_mask_tensor], dim=1),
        }
        discriminator_data = {
            'Vis': vis_tensor,
            'Inf': inf_tensor,
        }

        with torch.no_grad():
            generator_feats, _, _ = gan_model(discriminator_data, generator_data)

        # Output is in normalized space; convert back to image space before saving.
        fused = generator_feats['Generator_1'][0].detach().cpu()
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        fused = fused * std_t + mean_t
        fused = fused.clamp(0, 1)

        fused_img = to_pil_image(fused)
        save_path = os.path.join(out_dir, fname)
        fused_img.save(save_path)
        print(f"[ok] saved fused image: {fname}")
