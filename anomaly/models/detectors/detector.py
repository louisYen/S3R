import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')

from torch import Tensor
from einops import rearrange

from anomaly.models.modules import deNormal, GlobalStatistics
from anomaly.models.modules import enNormal


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.value = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0) # value

        if bn_layer:
            self.alter = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            ) # output
            nn.init.constant_(self.alter[1].weight, 0)
            nn.init.constant_(self.alter[1].bias, 0)
        else:
            self.alter = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.alter.weight, 0)
            nn.init.constant_(self.alter.bias, 0)

        self.query = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0) # query

        self.key = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0) # key

        if sub_sample: # default = False
            self.value = nn.Sequential(self.value, max_pool_layer)
            self.key = nn.Sequential(self.key, max_pool_layer)

    def forward(self, x: Tensor, return_nl_map: bool=False):
        """
        :param x: BCT
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        identity = x

        B, C, T = x.shape
        D = self.inter_channels

        value = self.value(x).view(B, D, -1) # BDT
        value = value.transpose(-2, -1) # BTD

        query = self.query(x).view(B, D, -1) # BDT
        query = query.transpose(-2, -1) # BTD
        key = self.key(x).view(B, D, -1) # BDT

        attn = query @ key # BTT
        attn = attn / T

        out = torch.matmul(attn, value)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(B, self.inter_channels, *x.size()[2:])
        out = self.alter(out)
        out = out + identity

        if return_nl_map:
            return out, attn
        return out


class NonLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Aggregate(nn.Module):
    def __init__(
        self,
        dim: int=2048,
        reduction: int=4,
    ):
        super(Aggregate, self).__init__()

        dim_inner = dim // reduction

        bn = nn.BatchNorm1d
        self.dim = dim

        self.conv_1 = nn.Sequential(
            nn.Conv1d(dim, dim_inner, kernel_size = 3,
                stride = 1, dilation = 1, padding = 1),
            nn.GroupNorm(num_groups = 8, num_channels = dim_inner, eps=1e-05),
            nn.ReLU())
        self.conv_2 = nn.Sequential(
            nn.Conv1d(dim, dim_inner, kernel_size=3,
                stride = 1, dilation = 2, padding = 2),
            nn.GroupNorm(num_groups = 8, num_channels = dim_inner, eps=1e-05),
            nn.ReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv1d(dim, dim_inner, kernel_size = 3,
                stride = 1, dilation = 4, padding = 4),
            nn.GroupNorm(num_groups = 8, num_channels = dim_inner, eps=1e-05),
            nn.ReLU())
        self.conv_4 = nn.Sequential(
            nn.Conv1d(dim, dim_inner, kernel_size = 1,
                stride = 1, padding = 0, bias = False),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = 3,
                stride = 1, padding = 1, bias = False), # TODO: should we keep the bias?
            nn.GroupNorm(num_groups = 8, num_channels = dim, eps=1e-05),
            nn.ReLU())

        self.non_local = NonLocalBlock1D(dim_inner, sub_sample=False, bn_layer=True)


    def forward(self, x: Tensor):

        x: Tensor # input feature of shape BTC

        out = x.transpose(-2, -1) # BCT
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)

        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim = 1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)   # fuse all the features together
        out = out + residual # BCT

        out = out.transpose(-2, -1) # BTC

        return out


class S3R(nn.Module):
    """ S3R Model """

    def __init__(
        self,
        dim: int = 2048,
        batch_size: int = 32,
        quantize_size: int = 32,
        dropout: float = 0.7,
        modality:str = 'univ-task',
    ):

        super(S3R, self).__init__()
        self.batch_size: int = batch_size
        self.k_anomaly: int = quantize_size // 10 # 3
        self.k_regular: int = quantize_size // 10 # 3

        self.video_embedding = nn.Sequential(
            Aggregate(dim),
            nn.Dropout(dropout))
        self.macro_embedding = nn.Sequential(
            Aggregate(dim),
            nn.Dropout(dropout))

        self.en_normal = enNormal(dim, modality=modality)
        self.de_normal = deNormal(dim, dim // 2, reduction=16)

        self.video_projection = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-05),
            nn.ReLU())
        self.macro_projection = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding = 1),
            nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-05),
            nn.ReLU())

        self.video_classifier = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 16, 1),
            nn.Sigmoid())
        macro_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 16, 1))
        self.macro_classifier = GlobalStatistics(mlp=macro_mlp)

        self.drop_out = nn.Dropout(dropout)
        self.apply(weight_init)

    def forward(
            self,
            video: Tensor, # input video of shape BNTC, N=num_crops (default=10), C=2048
            macro: Tensor, # dictionary of shape BSTC, C=2048
        ):

        device = video.device

        k_anomaly = self.k_anomaly
        k_regular = self.k_regular

        B, N, T, C = video.shape

        video = rearrange(video, 'b n t c -> (b n) t c') # (BN)TC

        # ========
        # enNormal
        # --------
        macro, memory_attn = self.en_normal(video, macro)

        x_video = self.video_embedding(video) # (BN)TC
        x_macro = self.macro_embedding(macro) # (BN)TC

        # ========
        # deNormal
        # --------
        x_video, x_macro = self.de_normal(x_video, x_macro) # (BN)TC

        # ==========
        # classifier
        # ----------
        video_embeds = x_video
        video_scores = self.video_classifier(video_embeds) # (BN)T1
        video_scores = video_scores.view(B, N, -1).mean(1) # BNT
        video_scores = video_scores.unsqueeze(dim=2) # BT1

        macro_scores = self.macro_classifier(x_macro.transpose(1, 2)) # (BN)1
        macro_scores = macro_scores.contiguous().view(-1, N, 1) # BN1
        macro_scores = macro_scores.mean(dim = 1) # B1

        regular_videos = video_embeds[0:self.batch_size * N]
        regular_scores = video_scores[0:self.batch_size]

        anomaly_videos = video_embeds[self.batch_size * N:]
        anomaly_scores = video_scores[self.batch_size:]

        feat_magnitudes = torch.norm(video_embeds, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(B, N, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes

        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            anomaly_scores = regular_scores
            anomaly_videos = regular_videos

        select_idx = torch.ones_like(nfea_magnitudes).to(device)
        select_idx = self.drop_out(select_idx)

        # ========================================================
        # process abnormal videos -> select top3 feature magnitude
        # --------------------------------------------------------
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_anomaly, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, anomaly_videos.shape[2]])

        anomaly_videos = anomaly_videos.view(n_size, N, T, C)
        anomaly_videos = anomaly_videos.permute(1, 0, 2, 3)

        total_select_abn_feature = torch.zeros(0).to(device)
        for abnormal_feature in anomaly_videos:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, anomaly_scores.shape[2]])
        anomaly_score = torch.mean(torch.gather(anomaly_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude

        # ======================================================
        # process normal videos -> select top3 feature magnitude
        # ------------------------------------------------------

        select_idx_normal = torch.ones_like(nfea_magnitudes).to(device)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_regular, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, regular_videos.shape[2]])

        regular_videos = regular_videos.view(n_size, N, T, C)
        regular_videos = regular_videos.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0).to(device)
        for nor_fea in regular_videos:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, regular_scores.shape[2]])
        regular_score = torch.mean(torch.gather(regular_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feature_select_anomaly = total_select_abn_feature # NKC, K=topk (default = 3)
        feature_select_regular = total_select_nor_feature # NKC

        return dict(
            anomaly_score = anomaly_score,
            regular_score = regular_score,
            feature_select_anomaly = feature_select_anomaly,
            feature_select_regular = feature_select_regular,
            feature_abnormal_bottom = feature_select_anomaly,
            feature_select_normal_bottom = feature_select_anomaly,
            video_scores = video_scores,
            macro_scores = macro_scores,
            scores_normal_bottom = feature_select_anomaly,
            scores_normal_abnormal_bag = feature_select_anomaly,
            feature_magnitudes = feat_magnitudes,
            memory_attn = memory_attn)
