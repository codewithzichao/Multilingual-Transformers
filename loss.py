import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self,weight,prior,tau=1.0,mmi=False):
        super(MyLoss, self).__init__()

        self.tau=tau
        self.mmi=mmi
        self.celoss=nn.CrossEntropyLoss(weight=weight)
        self.prior=torch.log(prior+1e-8).unsqueeze(dim=0)

        self.lang_celoss=nn.CrossEntropyLoss()

    def forward(self,y_pred,y_true,lang_pred=None,lang_id=None):
        if self.mmi is True:
            y_pred=y_pred+self.tau*self.prior

        if lang_pred is not None:
            lang_loss=self.lang_celoss(lang_pred,lang_id)
            return self.celoss(y_pred,y_true)+0.5*lang_loss
        else:
            return self.celoss(y_pred,y_true)


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss