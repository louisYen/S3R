
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalStatistics(nn.Module):
    def __init__(
            self,
            mlp = None,
            pool_types: list = ['avg', ]
        ):
        super(GlobalStatistics, self).__init__()

        self.flat = Flatten()
        if mlp is not None:
            self.mlp = mlp

        self.pool_types = pool_types

    def forward(self, input):
        _, c_input, t_input = input.size()

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool1d(input, t_input, stride = t_input)
                avg_pool = self.flat(avg_pool)

                channel_att_raw = self.mlp(avg_pool)

            elif pool_type == 'max':
                max_pool = F.max_pool1d(input, t_input, stride = t_input)
                max_pool = self.flat(max_pool)

                channel_att_raw = self.mlp(max_pool)

            elif pool_type == 'lp':
                lp_pool = F.lp_pool1d(input, 2, t_input, stride = t_input)
                lp_pool = self.flat(lp_pool)

                channel_att_raw = self.mlp(lp_pool)

            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_1d(input)
                lse_pool = self.flat(lse_pool)

                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        logit = torch.sigmoid(channel_att_sum)

        return logit

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(
            self,
            gate_channels: int,
            reduction_ratio: int = 16,
            pool_types: list = ['avg', 'max']
        ):
        super(ChannelGate, self).__init__()

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))

        self.pool_types = pool_types

    def forward(self, video, macro):
        """
        :param video
            - size: [bs * n, c, t]
        :param macro
            - size: [bs * n[repeated], c, t]
        """

        _, c_video, t_video = video.size()
        _, c_macro, t_macro = macro.size()

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                video_avg_pool = F.avg_pool1d(video, t_video, stride = t_video)
                macro_avg_pool = F.avg_pool1d(macro, t_macro, stride = t_macro)

                # >> residual
                avg_pool = video_avg_pool - macro_avg_pool

                # # TODO abs
                # avg_pool = torch.abs(avg_pool)

                channel_att_raw = self.mlp(avg_pool)

            elif pool_type == 'max':
                video_max_pool = F.max_pool1d(video, t_video, stride = t_video)
                macro_max_pool = F.max_pool1d(macro, t_macro, stride = t_macro)

                # >> residual
                max_pool = video_max_pool - macro_max_pool

                channel_att_raw = self.mlp(max_pool)

            elif pool_type == 'lp':
                video_lp_pool = F.lp_pool1d(video, 2, t_video, stride = t_video)
                macro_lp_pool = F.lp_pool1d(macro, 2, t_macro, stride = t_macro)

                # >> residual
                lp_pool = video_lp_pool - macro_lp_pool
                channel_att_raw = self.mlp(lp_pool)

            elif pool_type == 'lse':
                # LSE pool only
                video_lse_pool = logsumexp_1d(video)
                macro_lse_pool = logsumexp_1d(macro)

                # >> residual
                lse_pool = video_lse_pool - macro_lse_pool
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        weight = torch.sigmoid(channel_att_sum).unsqueeze(2)

        return weight

class ChannelAttention(nn.Module):
    def __init__(
            self,
            dim_input: int,
            reduction: int = 16,
            pool_types: list = ['avg', ],
        ):
        super(ChannelAttention, self).__init__()

        self.channel_gate = ChannelGate(dim_input, reduction, pool_types)

    def forward(self, video, macro):
        """
        :param video, macro
            - size: [bs * n, c, t]
        """

        scale = self.channel_gate(video, macro) # attention of different actions

        video = video * scale.expand_as(video) # excite different actions
        macro = macro * (1.- scale).expand_as(macro) # excite same actions

        return video, macro

class deNormal(nn.Module):
    def __init__(
            self,
            dim_input: int = 2048,
            dim_inner: int = 1024,
            reduction: int = 16,
            pool_types: list = ['avg', ],
            temporal_last: bool = False,
        ):
        super(deNormal, self).__init__()

        self.temporal_last = temporal_last

        self.channel_attention = ChannelAttention(dim_input, reduction, pool_types)

    def forward(self, video, macro):
        """
        :param video, macro of shape (BxN)xTxC
        """

        if not self.temporal_last:
            video, macro = video.transpose(1, 2), macro.transpose(1, 2) # (BN)CT

        # channel-wise attention
        video, macro = self.channel_attention(video, macro)

        if not self.temporal_last:
            video, macro = video.transpose(1, 2), macro.transpose(1, 2) # (BN)CT

        return video, macro
