import torch
import torch.nn as nn
import torch.fft as fft

class AttentionFusion(nn.Module):
    def __init__(self, seq_length, input_dim, total_segments):
        super().__init__()
        self.total_segments = total_segments
        self.input_dim = input_dim

        # 权重网络
        self.weight_net = nn.ModuleList([nn.Sequential(
            nn.Linear(seq_length * input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        ) for _ in range(total_segments)])

        # 原始特征和融合特征的进一步融合模块
        self.before_after_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # 输入维度是 input_dim * 2
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),  # 输出维度是 input_dim
            nn.Dropout(p=0.1),
            nn.ReLU(),
        )
    
    def get_frequency_feature(self, modality, total_segments):
        modality_fft = fft.fft(modality, dim=-1)
        modality_fft = fft.fftshift(modality_fft, dim=-1)

        length = modality_fft.shape[-1]
        mid = length // 2
        step = length // (2 * total_segments)

        mask = torch.zeros_like(modality_fft).to(modality.device)  # 使用传入张量的设备

        # 生成从低到高的全部段数
        frequency_segments = []
        for t in range(1, total_segments + 1):
            if t == 1:
                mask[:, :, mid - step:mid + step] = 1
            else:
                mask[:, :, mid - (t * step):mid - ((t - 1) * step)] = 1
                mask[:, :, mid + ((t - 1) * step):mid + (t * step)] = 1

            x_freq = modality_fft.clone() * mask.clone()  # 显式克隆避免原地操作

            # IFFT
            x_freq = fft.ifftshift(x_freq, dim=(-1))
            x_filtered = fft.ifftn(x_freq, dim=(-1)).real

            frequency_segments.append(x_filtered)

            # 重置mask
            mask.fill_(0)

        return frequency_segments

    def forward(self, ori_sequence):
        unsqueeze = len(ori_sequence.shape) == 2
        if unsqueeze:
            ori_sequence = ori_sequence.unsqueeze(-1).clone()  # 使用clone避免原地操作

        frequency_list = self.get_frequency_feature(ori_sequence, self.total_segments)

        weights = []
        for i in range(len(frequency_list)):
            frequency_feature = frequency_list[i].reshape(frequency_list[i].shape[0], -1)
            weight = self.weight_net[i](frequency_feature)
            weights.append(weight)
        weights = torch.cat(weights, dim=-1)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        fusion_output = torch.zeros_like(frequency_list[0])
        for i in range(len(frequency_list)):
            fusion_output = fusion_output + frequency_list[i] * weights[:, i:i + 1].unsqueeze(-1)  # 避免原地操作

        combined_features = torch.cat([ori_sequence, fusion_output], dim=-1)
        combined_features = self.before_after_fusion(combined_features)

        if unsqueeze:
            combined_features = combined_features.squeeze(-1)
        return combined_features