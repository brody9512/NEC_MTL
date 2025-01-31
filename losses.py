from typing import Optional, List
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch
import segmentation_models_pytorch as smp

# from segmentation_models_pytorch.utils.base import to_tensor
# To_tensor is a helper function from old SMP library, but it's not exposed on newer versions 
# So, create custom function to onvert Python lists/NumPy arrays into PyTorch tensors
def to_tensor(data, dtype=None):
    return torch.as_tensor(data, dtype=dtype)

# Same for both train & test
def soft_dice_score(output, target, smooth=0.0, eps=1e-7, dims=None): # (output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None,) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality  = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality  = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class DiceLoss(_Loss): # (nn.module)
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        assert mode in {'binary', 'multilabel', 'multiclass'}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != 'binary', "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long) ## where is this function from?

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

class Dice_BCE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = 1.0
        self.bce_weight  = 1.0

    def forward(self, y_pred, y_true):
        dloss = self.dice_loss(y_pred, y_true)
        bloss = self.bce_loss(y_pred, y_true)
        return self.dice_weight*dloss + self.bce_weight*bloss

class Uptask_Loss_Train(torch.nn.Module):
    def __init__(self, cls_weight=1.0, seg_weight=1.0, loss_type='bc_di'):
        
        super().__init__()
        
        # Initialize loss functions as None
        self.loss_cls = None
        self.loss_seg = None
        self.loss_rec = torch.nn.L1Loss()
        
        # Weights for each component of the loss
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.rec_weight = 1.0
        
        # Select loss type
        self.loss_type = loss_type
        
        self.loss_cls = torch.nn.BCEWithLogitsLoss()
        self.loss_seg = Dice_BCE_Loss()


    def forward(self, cls_pred=None, seg_pred=None, cls_gt=None, seg_gt=None): # ,consist=False
        # If classification prediction and ground truth are provided, calculate classification loss
        loss_cls = self.loss_cls(cls_pred, cls_gt) if cls_pred is not None and cls_gt is not None else 0
        
        # If segmentation prediction and ground truth are provided, calculate segmentation loss
        loss_seg = self.loss_seg(seg_pred, seg_gt) if seg_pred is not None and seg_gt is not None else 0
        
        # Combine the losses
        total = self.cls_weight * loss_cls + self.seg_weight * loss_seg
        
        # Record the individual components of the loss
        total_ = {'CLS_Loss': (self.cls_weight * loss_cls).item(), 'SEG_Loss': (self.seg_weight * loss_seg).item()}
        
        return total, total_


class Uptask_Loss_Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls     = torch.nn.BCEWithLogitsLoss()
        self.loss_seg     = Dice_BCE_Loss()
        self.loss_rec     = torch.nn.L1Loss()

    def forward(self, cls_pred=None, cls_gt=None):
            total = self.loss_cls(cls_pred, cls_gt)
            return total