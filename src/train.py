import click
import numpy as np
from torch.utils.data import random_split, DataLoader

from dataloader.dataloader import LoadDataSet
from dataloader.transform import get_train_transform
from utils.utils import load_yaml


@click.command()
@click.option("--cfg")

def main(cfg: str):
    cfg = load_yaml(cfg)

    train_dataset = LoadDataSet(
        cfg=cfg,
        transform=get_train_transform(cfg)
    )
    
    split_ratio = cfg["TRAIN"]["INFO"]["SPLIT_RATIO"]
    
    train_size = int(np.round(train_dataset.__len__() * (1 - split_ratio), 0))
    val_size = int(np.round(train_dataset.__len__() * split_ratio, 0))
    train_data, val_data = random_split(
        train_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=10,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=10
    )

if __name__ == "__main__":
    main()