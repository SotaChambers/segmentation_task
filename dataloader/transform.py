import  albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose


def get_train_transform(cfg) -> Compose:
    """学習用の前処理．dataloaderで抽出された画像は学習の前に本処理が実行される．

    args:
        cfg: 設定ファイル

    Returns:
        Compose: 前処理一式
    """
    resize_val = cfg["TRAIN"]["TRANSFORM"]["RESIZE"]
    h_frop = cfg["TRAIN"]["TRANSFORM"]["H_FRIP"]
    v_frop = cfg["TRAIN"]["TRANSFORM"]["V_FRIP"]

    transform = [
        A.Resize(resize_val, resize_val), #リサイズ
        # (img - mean * 255) / (std * 255)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=h_frop), #水平方向のフリップ
        A.VerticalFlip(p=v_frop), #垂直方向のフリップ
        ToTensorV2()
    ]
    return Compose(transform)