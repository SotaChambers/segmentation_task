import click
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

from dataloader.dataloader import LoadDataSet
from dataloader.transform import get_train_transform
from utils.utils import load_yaml, save_ckp
from model.dice_loss import DiceLoss
from model.unet import UNet
from model.iou import IoU


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

    # Datasetの作成
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=10,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=10
    )

    # 各インスタンスの作成
    model = UNet(3, 1).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceLoss()
    accuracy_metric = IoU()
    num_epochs = cfg["TRAIN"]["INFO"]["EPOCH"]
    valid_loss_min = np.inf

    checkpoint_path = 'model/chkpoint_'
    best_model_path = 'model/bestmodel.pt'

    total_train_loss = []
    total_train_score = []
    total_valid_loss = []
    total_valid_score = []
    losses_value = 0

    for epoch in range(num_epochs):
        train_loss = []
        train_score = []
        valid_loss = []
        valid_score = []
        pbar = tqdm(train_loader, desc = 'description')
        # トレーニング
        for x_train, y_train in pbar:
            x_train = torch.autograd.Variable(x_train).cpu()
            y_train = torch.autograd.Variable(y_train).cpu()
            optimizer.zero_grad()
            output = model(x_train)
            # 損失計算
            loss = criterion(output, y_train)
            losses_value = loss.item()
            # 精度評価
            score = accuracy_metric(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss.append(losses_value)
            train_score.append(score.item())
            pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")

        # 評価
        with torch.no_grad():
            for image,mask in val_loader:
                image = torch.autograd.Variable(image).cpu()
                mask = torch.autograd.Variable(mask).cpu()
                output = model(image)
                ## 損失計算
                loss = criterion(output, mask)
                losses_value = loss.item()
                ## 精度評価
                score = accuracy_metric(output,mask)
                valid_loss.append(losses_value)
                valid_score.append(score.item())

        total_train_loss.append(np.mean(train_loss))
        total_train_score.append(np.mean(train_score))
        total_valid_loss.append(np.mean(valid_loss))
        total_valid_score.append(np.mean(valid_score))
        print(f"Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}")
        print(f"Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}")

        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': total_valid_loss[-1],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    
        # checkpointの保存
        save_ckp(checkpoint, False, checkpoint_path, best_model_path, epoch)


if __name__ == "__main__":
    main()