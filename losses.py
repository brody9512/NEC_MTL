# good. just some args are in the function which might need to be fixed later on, but the skeleton/structure looks good.
from typing import Optional, List
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch

# from MTL train
def soft_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None,) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class DiceLoss(_Loss):
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
            classes = to_tensor(classes, dtype=torch.long)

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

class Dice_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function_1 = DiceLoss(mode='binary', from_logits=True)
        self.loss_function_2 = torch.nn.BCEWithLogitsLoss()
        self.dice_weight     = 1.0   
        self.bce_weight      = 1.0   

    def forward(self, y_pred, y_true):
        dice_loss  = self.loss_function_1(y_pred, y_true)
        bce_loss   = self.loss_function_2(y_pred, y_true)

        return self.dice_weight*dice_loss + self.bce_weight*bce_loss

class Consistency_Loss_Train(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        k_size=k_size_ ## 만액에 size 1024에 k_size= 16이라면!
        final_size = int((min_side_ / k_size** 2))**2  #root 가하는 건 math.sqrt(min_side_)
        self.L2_loss  = torch.nn.MSELoss()
        self.maxpool  = torch.nn.MaxPool2d(kernel_size=k_size, stride=k_size, padding=0)
        self.avgpool  = torch.nn.AvgPool2d(kernel_size=k_size, stride=k_size, padding=0)
        self.fc = nn.Linear(1,final_size)  # 채널 수를 1에서 16로 변경 
        #min_side =1024 , kernel_size = 8    final_size>> 256 
        #min_side =1024 , kernel_size = 16   final_size>> 16

    def forward(self, y_cls, y_seg):
        
        y_cls = torch.sigmoid(y_cls)  # (B, C)
        y_seg = torch.sigmoid(y_seg)  # (B, C, H, W)
        
        y_cls = self.fc(y_cls)

        y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)  # (B, C)
        
        loss  = self.L2_loss(y_seg, y_cls)

        return loss
    
class Consistency_Loss_Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L2_loss  = torch.nn.MSELoss()
        self.maxpool  = torch.nn.MaxPool2d(kernel_size=16, stride=16, padding=0)
        self.avgpool  = torch.nn.AvgPool2d(kernel_size=16, stride=16, padding=0)

    def forward(self, y_cls, y_seg):
        y_cls = torch.sigmoid(y_cls)  # (B, C)
        y_seg = torch.sigmoid(y_seg)  # (B, C, H, W)

        y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)  # (B, C)
        loss  = self.L2_loss(y_seg, y_cls)

        return loss

class Uptask_Loss_Train(torch.nn.Module):
    def __init__(self, cls_weight=1.0, seg_weight=1.0, consist_weight=0, loss_type='bc_di'):
        
        super().__init__()
        
        # Initialize loss functions as None
        self.loss_cls = None
        self.loss_seg = None
        self.loss_rec = torch.nn.L1Loss()
        self.loss_consist = Consistency_Loss_Train()
        
        # Weights for each component of the loss
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.rec_weight = 1.0
        self.consist_weight = consist_weight
        
        # Select loss type
        self.loss_type = loss_type
        
        self.loss_cls = torch.nn.BCEWithLogitsLoss()
        self.loss_seg = Dice_BCE_Loss()
        
        '''
        Can this be removed and simplfied since we're going with bc_di and disregarding others?
        if self.loss_type == 'bc_di':
            self.loss_cls = torch.nn.BCEWithLogitsLoss()
            self.loss_seg = Dice_BCE_Loss()

        elif self.loss_type == 'bc_iou':
            self.loss_cls = torch.nn.BCEWithLogitsLoss()  # Assuming we keep BCE for classification
            self.loss_seg = IoULoss()

        elif self.loss_type == 'bc_tv':
            self.loss_cls = torch.nn.BCEWithLogitsLoss()
            self.loss_seg = TverskyLoss()
            
        elif self.loss_type == 'fo_di':
            self.loss_cls = FocalLoss()
            self.loss_seg = Dice_BCE_Loss()
            
        elif self.loss_type == 'fo_tv':
            self.loss_cls = FocalLoss()
            self.loss_seg = TverskyLoss()

        elif self.loss_type == 'fo_iou':
            self.loss_cls = FocalLoss()
            self.loss_seg = IoULoss()
        '''


    def forward(self, cls_pred=None, seg_pred=None, rec_pred=None, cls_gt=None, seg_gt=None, rec_gt=None, consist=False):
        # If classification prediction and ground truth are provided, calculate classification loss
        loss_cls = self.loss_cls(cls_pred, cls_gt) if cls_pred is not None and cls_gt is not None else 0
        
        # If segmentation prediction and ground truth are provided, calculate segmentation loss
        loss_seg = self.loss_seg(seg_pred, seg_gt) if seg_pred is not None and seg_gt is not None else 0
        
        # If consistency needs to be calculated
        loss_consist = self.loss_consist(cls_pred, seg_pred) if consist else 0
        
        # Combine the losses
        total = self.cls_weight * loss_cls + self.seg_weight * loss_seg + self.consist_weight * loss_consist
        
        # Record the individual components of the loss
        total_ = {'CLS_Loss': (self.cls_weight * loss_cls).item(), 'SEG_Loss': (self.seg_weight * loss_seg).item()}
        if consist:
            total_['Consist_Loss'] = (self.consist_weight * loss_consist).item()
        
        return total, total_


class Uptask_Loss_Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls     = torch.nn.BCEWithLogitsLoss()
        self.loss_seg     = Dice_BCE_Loss()
        self.loss_rec     = torch.nn.L1Loss()
        self.loss_consist = Consistency_Loss_Test()


    def forward(self, cls_pred=None, cls_gt=None):
            total = self.loss_cls(cls_pred, cls_gt)
            return total


