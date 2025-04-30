import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 导入Config的正确方式
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class HighResolutionBranch(nn.Module):
    """
    高分辨率特征处理分支 (处理CQT特征)
    """
    def __init__(self, channels=[64, 128, 256], kernels=[3, 3, 3]):
        super(HighResolutionBranch, self).__init__()
        
        layers = []
        in_channels = 1  # 输入为单通道CQT
        
        # 添加卷积层
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernels)):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 计算输出特征维度 (用于后续全连接层)
        self.output_dim = channels[-1]
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入CQT特征 [batch_size, 1, freq_bins, time_frames]
        Returns:
            features: 提取的特征 [batch_size, output_dim]
        """
        # 批量为1时切换BN为评估模式
        if x.size(0) == 1:
            # 保存当前模式
            training_mode = self.training
            # 临时切换为评估模式
            self.eval()
            # 执行前向传播
            features = self.feature_extractor(x)
            # 恢复原始模式
            if training_mode:
                self.train()
        else:
            # 正常前向传播
            features = self.feature_extractor(x)
        
        # 全局平均池化 (去除频率和时间维度)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # 展平
        features = features.view(features.size(0), -1)
        
        return features


class TimbreBranch(nn.Module):
    """
    音色特征处理分支
    """
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(TimbreBranch, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # 添加全连接层
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=Config.DROPOUT_RATE))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出维度
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入音色特征 [batch_size, input_dim]
        Returns:
            features: 处理后的特征 [batch_size, output_dim]
        """
        # 批量为1时切换BN为评估模式
        if x.size(0) == 1:
            # 保存当前模式
            training_mode = self.training
            # 临时切换为评估模式
            self.eval()
            # 执行前向传播
            features = self.feature_extractor(x)
            # 恢复原始模式
            if training_mode:
                self.train()
            return features
        else:
            # 正常前向传播
            return self.feature_extractor(x)


class TGCritic(nn.Module):
    """
    TG-CRITIC模型 - 结合时域和频域特征的声音质量评估模型
    """
    def __init__(self, num_classes=4):
        super(TGCritic, self).__init__()
        
        # CQT特征处理分支
        self.cqt_branch = HighResolutionBranch(
            channels=Config.CQT_CHANNELS,
            kernels=Config.CQT_KERNELS
        )
        
        # 音色特征处理分支
        self.timbre_branch = TimbreBranch(
            input_dim=Config.TIMBRE_DIM,
            hidden_dims=Config.TIMBRE_LAYERS
        )
        
        # 特征融合层
        combined_dim = self.cqt_branch.output_dim + self.timbre_branch.output_dim
        
        # 定义融合后的全连接层
        fc_layers = []
        dims = [combined_dim] + Config.FUSION_LAYERS
        
        for i in range(len(dims) - 1):
            fc_layers.append(nn.Linear(dims[i], dims[i+1]))
            fc_layers.append(nn.BatchNorm1d(dims[i+1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=Config.DROPOUT_RATE))
        
        # 添加输出层
        if Config.TASK_TYPE == 'classification':
            fc_layers.append(nn.Linear(dims[-1], num_classes))
        else:  # 回归任务
            fc_layers.append(nn.Linear(dims[-1], 1))
            fc_layers.append(nn.Tanh())  # 回归任务使用Tanh激活函数将输出限制在[-1, 1]
        
        self.fc = nn.Sequential(*fc_layers)
        
        # 确保所有参数使用相同的数据类型
        self.to(torch.float32)
    
    def forward(self, cqt, timbre):
        """
        前向传播
        Args:
            cqt: CQT特征 [batch_size, 1, freq_bins, time_frames]
            timbre: 音色特征 [batch_size, timbre_dim]
        Returns:
            outputs: 模型输出 [batch_size, num_classes]
        """
        # 确保输入数据类型一致
        cqt = cqt.to(torch.float32)
        timbre = timbre.to(torch.float32)
        
        # 特征提取
        cqt_features = self.cqt_branch(cqt)
        timbre_features = self.timbre_branch(timbre)
        
        # 特征融合
        combined_features = torch.cat([cqt_features, timbre_features], dim=1)
        
        # 批量为1时切换BN为评估模式
        if combined_features.size(0) == 1:
            # 保存当前模式
            training_mode = self.training
            # 临时切换为评估模式
            self.eval()
            # 执行前向传播
            outputs = self.fc(combined_features)
            # 恢复原始模式
            if training_mode:
                self.train()
        else:
            # 正常分类/回归
            outputs = self.fc(combined_features)
        
        # 验证输出维度是否符合任务类型
        num_classes = 1 if Config.TASK_TYPE == 'binary' else outputs.shape[1]
        # 检查线性层的输出维度而不是Tanh层
        if hasattr(self.fc[-1], 'out_features'):
            assert self.fc[-1].out_features == (1 if Config.TASK_TYPE == 'binary' else num_classes), \
                "输出层维度与任务类型不匹配"
        elif len(self.fc) > 1 and hasattr(self.fc[-2], 'out_features'):
            assert self.fc[-2].out_features == (1 if Config.TASK_TYPE == 'binary' else num_classes), \
                "输出层维度与任务类型不匹配"
        
        return outputs
