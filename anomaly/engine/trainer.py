import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

from torch.nn import L1Loss, MSELoss, Sigmoid
from anomaly.losses import SigmoidMAELoss, sparsity_loss, smooth_loss

class RTFM_loss(nn.Module):
    def __init__(
            self,
            alpha,
            margin
        ):
        super(RTFM_loss, self).__init__()

        self.alpha = alpha
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = nn.BCELoss()

    def forward(
            self,
            regular_score, # (regular) snippet-level classification score of shape Bx1
            anomaly_score, # (anomaly) snippet-level classification score of shape Bx1
            regular_label, # (regular) video-level label of shape B
            anomaly_label, # (anomaly) video-level label of shape B
            regular_crest, # (regular) selected top snippet features of shape (Bxn)xtopkxC
            anomaly_crest, # (anomaly) selected top snippet features of shape (Bxn)xtopkxC
        ):

        label = torch.cat((regular_label, anomaly_label), 0)
        anomaly_score = anomaly_score
        regular_score = regular_score

        score = torch.cat((regular_score, anomaly_score), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_anomaly = torch.abs(
                self.margin - torch.norm(torch.mean(anomaly_crest, dim=1),
                p=2, dim=1)) # Txn
        loss_regular = torch.norm(torch.mean(regular_crest, dim=1), p=2, dim=1) # Txn
        loss = torch.mean((loss_anomaly + loss_regular) ** 2)

        loss_total = loss_cls + self.alpha * loss

        return loss_total

class MacroLoss(nn.Module):
    def __init__(
            self,
        ):
        super(MacroLoss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, input, label):

        input = input.squeeze()
        target = torch.cat((label, label), dim = 0).cuda()

        loss = self.loss(input, target)

        return loss

def do_train(regular_loader, anomaly_loader, model, batch_size, optimizer, device):
    with torch.set_grad_enabled(True):
        model.train()

        """
        :param regular_video, anomaly_video
            - size: [bs, n=10, t=32, c=2048]
        """

        regular_video, regular_label, macro_video, macro_label = next(regular_loader)
        anomaly_video, anomaly_label, macro_video, macro_label = next(anomaly_loader)

        video = torch.cat((regular_video, anomaly_video), 0).to(device)
        macro = torch.cat((macro_video, macro_video), 0).to(device)

        outputs = model(video, macro)

        # >> parse outputs
        anomaly_score = outputs['anomaly_score']
        regular_score = outputs['regular_score']
        anomaly_crest = outputs['feature_select_anomaly']
        regular_crest = outputs['feature_select_regular']
        video_scores = outputs['video_scores']
        macro_scores = outputs['macro_scores']

        video_scores = video_scores.view(batch_size * 32 * 2, -1)

        video_scores = video_scores.squeeze()
        abn_scores = video_scores[batch_size * 32:]

        regular_label = regular_label[0:batch_size]
        anomaly_label = anomaly_label[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_magnitude = loss_criterion(
            regular_score,
            anomaly_score,
            regular_label,
            anomaly_label,
            regular_crest,
            anomaly_crest)

        loss_sparse = sparsity_loss(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth_loss(abn_scores, 8e-4)

        macro_criterion = MacroLoss()
        loss_macro = macro_criterion(macro_scores, macro_label)

        cost = loss_magnitude + loss_smooth + loss_sparse + loss_macro

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return cost


