import glob
import os
import random

from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

data_types_image = {'.png', '.jpg', '.jpeg', '.bmp'}
data_types_saved = {'.npy'}

data_types = data_types_image | data_types_saved

class MedVerseDataset(Dataset):
    def __init__(self, root_dir, shape=224):
        self.root_dir = root_dir

        self.shape = shape

        self.filenames = [
            f for f in glob.glob(os.path.join(root_dir, "*", "*.*"))
            if os.path.splitext(f)[1].lower() in data_types
        ]

    def prepare_img(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext in data_types_image:

            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {filename}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr = img.astype(np.float32) / 255.0  # 归一化到0-1
            arr = arr.transpose(2, 0, 1)  # 转成 (C,H,W)

            return arr
        
        elif ext in data_types_saved:
            arr = np.load(filename).astype(np.float32)
            if arr.ndim == 2:  # (H, W)
                arr = np.expand_dims(arr, axis=0)  # (1, H, W)
                arr = np.repeat(arr, 3, axis=0)  # (3, H, W)
            else:
                c_axis = int(np.argmin(arr.shape))
                if c_axis != 0:
                    # 将 channel 维度移到第一维
                    axes = [c_axis] + [i for i in range(len(arr.shape)) if i != c_axis]
                    arr = np.transpose(arr, axes)

                # 检查 channel 数量
                C = arr.shape[0]
                if C == 1:
                    arr = np.repeat(arr, 3, axis=0)  # 扩展为 3 通道
                elif C != 3:
                    raise ValueError(f"Unsupported channel size: {C}")
                
                
                # 检查动态范围，如果不在0-1，就minmax归一化
                arr_min, arr_max = arr.min(), arr.max()
                if arr_min < 0 or arr_max > 1:
                    arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)

            return arr
        else:
            raise ValueError(f"Unsupported file type: {ext}")


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = self.prepare_img(filename)
        img = self.RandomResizedCrop(img.transpose(1, 2, 0)).transpose(2, 0, 1)  # (C, H,W)
        return img

    def RandomResizedCrop(self, img, scale=(0.2, 1.0), ratio=(3/4, 4/3)):
        """模仿 torchvision.transforms.RandomResizedCrop"""
        H, W, C = img.shape
        area = H * W

        # 自动调整scale下限，保证最小裁剪面积不小于目标输出面积的0.5倍
        min_crop_area = max(self.shape * self.shape * 0.5, area * scale[0])
        scale_low = min_crop_area / area
        scale_high = scale[1]
        scale_adj = (scale_low, scale_high)

        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        for _ in range(10):  # 尝试 10 次找到合适的裁剪框
            target_area = random.uniform(*scale_adj) * area
            aspect_ratio = np.exp(random.uniform(*log_ratio))

            h = int(round(np.sqrt(target_area / aspect_ratio)))
            w = int(round(np.sqrt(target_area * aspect_ratio)))

            if 0< h <= H and 0 < w <= W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)

                crop = img[top: top + h, left: left + w, :]
                # 判断裁剪区域是否接近全黑
                if crop.mean() < 0.05:
                    continue  # 跳过全黑区域

                img = img[top: top + h, left: left + w, :]
                img = cv2.resize(img, (self.shape, self.shape), interpolation=cv2.INTER_AREA)
                return img

        # 如果没找到合适裁剪框，就 fallback 到中心裁剪
        min_side = min(H, W)
        img = img[(H - min_side) // 2:(H + min_side) // 2,
                  (W - min_side) // 2:(W + min_side) // 2, :]
        img = cv2.resize(img, (self.shape, self.shape), interpolation=cv2.INTER_AREA)
        return img
        
    def __len__(self):
        return len(self.filenames)
