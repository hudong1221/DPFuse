# File: fusion_dataset_with_masks.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MultiModalDataset_WithMasks(Dataset):
    """
    一个“多模态（CT、MRI）+ 掩码”联合读取的 Dataset。
    对每一条样本，按照配置：
      data_root/
        CT/
          images/      # e.g. 0001.png, 0002.png, …
          masks/       # e.g. 0001_mask.png, 0002_mask.png, …
        MRI/
          images/      # e.g. 0001.png, 0002.png, …
          masks/       # e.g. 0001_mask.png, 0002_mask.png, …

    sensors: ["CT", "MRI"]
    每个 sensor 下的 images/ 与 masks/ 里文件须一一对应，命名规则保持一致（basename相同）。

    本类在初始化时：
      1. 扫描第一个 sensor（如 "CT"）的 images/，提取所有 basename 列表；
      2. __getitem__ 时，依次对每个 sensor 去分别加载 images/ 下的 RGB 图 和 masks/ 下的单通道灰度图，
         都做相同的 Resize，再分别 ToTensor + Normalize（图像）／ToTensor + 二值化（掩码）；
      3. 最后将“CT_img(3ch) + CT_mask(1ch) + MRI_img(3ch) + MRI_mask(1ch)”在通道维上拼接，
         返回一个形状为 `(3+1 + 3+1, H, W) = (8, H, W)` 的 tensor。
    """

    def __init__(self,
                 root_dir,              # 根目录, 形如 "D:/Projects/.../data"
                 sensors,               # 如 ["CT", "MRI"]
                 image_suffix=".png",   # 原图后缀
                 mask_suffix="_mask.png",# 掩码后缀
                 input_size=(256, 256), # 最终 Resize 大小
                 mean=(0.5, 0.5, 0.5),   # 用于 Normalize 的均值（RGB 三通道）
                 std=(0.5, 0.5, 0.5)     # 用于 Normalize 的标准差
                 ):
        super().__init__()
        self.root_dir   = root_dir
        self.sensors    = sensors
        self.img_suf    = image_suffix
        self.mask_suf   = mask_suffix
        self.input_size = input_size
        self.mean       = mean
        self.std        = std

        # 1. 在第一个 sensor 下（比如 "CT"）的 images/ 目录里扫一遍，得到所有文件 basename 列表
        base_sensor = sensors[0]
        images_dir_0 = os.path.join(root_dir, base_sensor, "images")
        if not os.path.isdir(images_dir_0):
            raise RuntimeError(f"[Err] 找不到目录: {images_dir_0}")

        all_files = os.listdir(images_dir_0)
        # 只保留后缀匹配的文件，提取 basename
        self.basenames = [
            os.path.splitext(f)[0]     # 例如 "0001"
            for f in all_files
            if f.lower().endswith(self.img_suf)
        ]
        if len(self.basenames) == 0:
            raise RuntimeError(f"[Err] 在 {images_dir_0} 下没有找到任何 {self.img_suf} 文件。")

        # 2. 检查：其他 sensor 下的 images/ 目录也必须存在且包含同名文件
        for sensor in sensors[1:]:
            images_dir = os.path.join(root_dir, sensor, "images")
            if not os.path.isdir(images_dir):
                raise RuntimeError(f"[Err] 模态({sensor}) 的 images/ 目录不存在: {images_dir}")
            for base in self.basenames:
                img_name = base + self.img_suf
                if not os.path.exists(os.path.join(images_dir, img_name)):
                    raise RuntimeError(f"[Err] 在 {images_dir} 下缺少图像文件: {img_name}")

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        """
        返回：一个张量 input_tensor，shape = ((3+1) * len(sensors), H, W)
        其中 (3, H, W) 是 RGB 图像，(1, H, W) 是二值化掩码。
        这里默认每个灰度／彩色图都先转成 RGB（3通道），再 Normalize。
        """

        base = self.basenames[idx]   # e.g. "0001"
        modal_tensors = []

        # 对每个 sensor，依次读图 + 掩码
        for sensor in self.sensors:
            # 1) 读原图 → 转 RGB
            img_path = os.path.join(self.root_dir, sensor, "images", base + self.img_suf)
            img = Image.open(img_path).convert("RGB")   # 强制做成 3 通道

            # 2) 读对应掩码 → 转灰度单通道
            mask_path = os.path.join(self.root_dir, sensor, "masks", base + self.mask_suf)
            if not os.path.exists(mask_path):
                raise RuntimeError(f"[Err] 找不到 {sensor}/masks/ 下的掩码: {base + self.mask_suf}")
            mask = Image.open(mask_path).convert("L")   # 单通道灰度

            # 3) Resize
            img  = img.resize(self.input_size, resample=Image.BILINEAR)
            mask = mask.resize(self.input_size, resample=Image.NEAREST)

            # 4) 图像 → Tensor → Normalize
            img_t = TF.to_tensor(img)                           # (3, H, W)，值域[0,1]
            img_t = TF.normalize(img_t, self.mean, self.std)    # 归一化

            # 5) 掩码 → Tensor → 二值化
            mask_t = TF.to_tensor(mask)                         # (1, H, W)，值域[0,1]
            mask_t = (mask_t > 0.5).float()                      # {0.0, 1.0}

            # 6) 把本模态的“图像 (3ch)”和“掩码 (1ch)”都 append 进列表
            modal_tensors.append(img_t)   # shape=(3,H,W)
            modal_tensors.append(mask_t)  # shape=(1,H,W)

        # 7) 最后在通道维 (dim=0) 上拼接，例如 sensors=["CT","MRI"] 时
        #    modal_tensors = [CT_img(3,H,W), CT_mask(1,H,W), MRI_img(3,H,W), MRI_mask(1,H,W)]
        #    torch.cat(..., dim=0)  → shape=(8, H, W)
        input_tensor = torch.cat(modal_tensors, dim=0)
        return input_tensor
