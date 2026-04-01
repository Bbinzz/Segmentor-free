import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from rotate_mask import rotate_mask

class dreameDataset(Dataset):
    def __init__(self, datajson, root, train_num=-1, mask_rotated=False, angle=0):
        # 检查 JSON 文件是否存在
        if not os.path.isfile(datajson):
            raise FileNotFoundError(f"JSON file not found: {datajson}")
        with open(datajson, 'rt') as f:
            self.data = json.load(f)
            if train_num > 0:
                self.data = sorted(self.data, key=lambda x: x['source'])[:train_num]

        self.root = root
        self.mask_rotated = mask_rotated
        self.angle = angle

    def __len__(self):
        return len(self.data)

    def _safe_read_image(self, path, flags=cv2.IMREAD_COLOR, resize=None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(path, flags)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        if resize is not None:
            img = cv2.resize(img, resize)
        return img

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item.get('source')
        target_filename = item.get('target')
        prompt = item.get('prompt', "")

        # 构建绝对路径，避免字符串拼接
        source_path = os.path.join(self.root, source_filename)
        target_path = os.path.join(self.root, target_filename)

        # 安全读取图像
        source_img = self._safe_read_image(source_path, flags=cv2.IMREAD_COLOR, resize=(512, 512))
        target_img = self._safe_read_image(target_path, flags=cv2.IMREAD_COLOR, resize=(512, 512))

        # Mask路径默认使用 source filename，可根据实际修改
        mask_path = source_path

        if self.mask_rotated:
            try:
                mask = rotate_mask(mask_path, self.angle)
                mask = cv2.resize(mask, (512, 512))
            except Exception as e:
                raise RuntimeError(f"Error rotating mask at {mask_path}: {str(e)}")
        else:
            mask = self._safe_read_image(mask_path, flags=cv2.IMREAD_UNCHANGED, resize=(512, 512))

        # 处理source为3通道灰度图（重复通道）
        if len(mask.shape) == 2:
            source = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        elif len(mask.shape) == 3 and mask.shape[2] == 1:
            source = cv2.cvtColor(mask[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif len(mask.shape) == 3 and mask.shape[2] == 3:
            source = mask
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # OpenCV BGR to RGB
        target = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Normalize
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return {
            'jpg': target,
            'txt': prompt,
            'hint': source,
            'mask': mask
        }
