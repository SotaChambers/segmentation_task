from torch import nn

class DiceLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True
    ):
        super(DiceLoss, self).__init__()


    def forward(self, inputs, targets, smooth=1):
        """誤差関数を計算．Diceは予測と正解が重なるほど値が大きいので，間違っている場合に値が大きくなるようにする．
        (ref: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook)

        Args:
            inputs (_type_): 予測値
            targets (_type_): 正解値
            smooth (int, optional): Defaults to 1.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # inputsとtargetsが同じ場合(True)を数える
        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice