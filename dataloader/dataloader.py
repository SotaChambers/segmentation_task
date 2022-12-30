import os
import numpy as np
import torch

from torch.utils.data import Dataset
from albumentations import Compose
from skimage import io, transform

class LoadDataSet(Dataset):
    """自作のDataLoader．

    Args:
        path (str): 学習データのパス
        transform (Compose): DataLoaderで読み込む前に行う処理
    """
    def __init__(
        self,
        cfg: dict,
        transform: Compose=None
    ):
        self.cfg = cfg
        self.path = self.cfg["TRAIN"]["INFO"]["INPUT_PATH"]
        self.folders = os.listdir(self.path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image_folder = os.path.join(
            self.path, self.folders[idx], "images/"
        )
        mask_folder = os.path.join(
            self.path, self.folders[idx], "masks/"
        )
        image_path = os.path.join(
            image_folder, os.listdir(image_folder)[0]
        )
        # 画像データの取得
        resize_val = self.cfg["TRAIN"]["TRANSFORM"]["RESIZE"]
        img = io.imread(image_path)[:, :, :3].astype("float32")
        img = transform.resize(img, (resize_val, resize_val))
        mask = self.get_mask(mask_folder, resize_val, resize_val).astype("float32")
        
        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        mask = mask.permute(2, 0, 1) # 軸を変換(numpyのtranform)
        return (img, mask)



    def get_mask(
        self,
        mask_folder: str,
        IMG_HEIGHT: int,
        IMG_WIDTH: int
    ) -> np.array:
        """複数のマスクを合体して1枚のマスクにする処理

        Args:
            mask_folder (str): マスク画像群のパス
            IMG_HEIGHT (int): リサイズ後の画像の高さ
            IMG_WIDTH (int): リサイズ後の画像の幅
        """
        mask = np.zeros(
            (IMG_HEIGHT, IMG_WIDTH, 1),
            dtype=np.bool_
        )
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(
                os.path.join(mask_folder, mask_)
            )
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1) # (height, width) -> (height, width, 1)
            mask = np.maximum(mask, mask_) # Baseのマスクとmask_の比較

        return mask