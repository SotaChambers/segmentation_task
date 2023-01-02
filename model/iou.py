from torch import nn


class IoU(nn.Module):
    def __init__(self, weight, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """ref: https://qiita.com/shoku-pan/items/35eae224c59989957623

        Args:
            inputs (_type_): _description_
            targets (_type_): _description_
            smooth (int, optional): _description_. Defaults to 1.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU