from torch import nn
import torch

class UNet(nn.Module):
    """セグメンテーションを行うモデルであるUNetクラス

    Args:
        input_channels (int): 入力のチャンネル数
        output_channels (int): 出力のチャンネル数
    """
    def __init__(
        self,
        input_channels,
        output_channels
    ):
        super().__init__()
        self.conv1 = conv_bn_relu(input_channels,64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, output_channels, 1)

        # 正規分布でパラメータを初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """順方向のプロセスを定義
        - Down Sampling
            input(x): (10, 3, 256, 256)
            -> x1: (10, 64, 256, 256)
            -> p1(pooling): (10, 64, 128, 128)
            -> x2: (10, 128, 128, 128)
            -> p2(pooling): (10, 128, 64, 64)
            -> x3: (10, 256, 64, 64) 
            -> p3(pooling): (10, 256, 32, 32)
            -> x4: (10, 512, 32, 32) 
            -> p4(pooling): (10, 512, 16, 16)
            -> x5: (10, 1024, 16, 16) 
        
        - Up Sampling
            x6: (10, 512, 32, 32)
            -> x6(skip connection): (10, 1024, 32, 32)
            -> x6: (10, 512, 32, 32)
            -> x7: (10, 256, 64, 64)
            -> x7(skip connection): (10, 512, 64, 64)
            -> x7: (10, 256, 64, 64)
            -> x8: (10, 128, 128, 128)
            -> x8(skip connection): (10, 256, 128, 128)
            -> x8: (10, 128, 128, 128)
            -> x9: (10, 64, 256, 256)
            -> x9(skip connection): (10, 128, 256, 256)
            -> x9: (10, 64, 256, 256)
            -> output: (10, 1, 256, 256)

        Args:
            x (torch.Tensor): 画像のテンソル

        Returns:
            torch.Tensort: 画像のテンソル
        """
        # 正規化
        x = x/255.

        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # 資料中の『Up Sampling』に当たる部分, torch.catによりSkip Connectionをしている
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.sigmoid(output)

        return output


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding : int = 1
) -> nn.Sequential:
    """UNetの層を1プロセスを定義
    
    - Conv2d
    畳み込みを行う

    - BatchNorm2d
    バッチ正規化を行う(ref: https://atmarkit.itmedia.co.jp/ait/articles/2011/06/news024_2.html)

    - ReLU
    活性化関数にReLUを適用

    Args:
        in_channels (int): 	入力のチャンネル数
        out_channels (int): 出力のチャンネル数
        kernel_size (int, optional): フィルタ（カーネル）のサイズ. Defaults to 3.
        stride (int, optional): フィルタを動かす幅. Defaults to 1.
        padding (int, optional): パディングの数. Defaults to 1.

    Returns:
        nn.Sequential: ニューラルネットワークの層
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def down_pooling():
    """Max Poolingを定義

    Returns:
        nn.MaxPool2d: PytorchのMaxPooling
    """
    return nn.MaxPool2d(2)


def up_pooling(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2
) -> nn.Sequential:
    """UNetの1プロセスを定義
    
    - ConvTranspose2d
    転置畳み込みを行う(ref: https://qiita.com/ToppaD/items/44eb081f0cdf1e2b08c1)

    Args:
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        kernel_size (int, optional): フィルタ（カーネル）のサイズ. Defaults to 2.
        stride (int, optional): フィルタを動かす幅. Defaults to 2.

    Returns:
        nn.Sequential: _description_
    """
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )