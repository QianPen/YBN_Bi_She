import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class VocalQualityDataset(Dataset):
    """声音质量评估数据集"""
    
    def __init__(self, data_dir, label_file):
        """
        初始化数据集
        Args:
            data_dir: 包含CQT和Timbre特征的目录
            label_file: 标签CSV文件路径
        """
        self.data_dir = data_dir
        
        # 读取标签文件
        self.labels_df = pd.read_csv(label_file)
        
        # 获取文件列表
        self.file_ids = self.labels_df['file_id'].tolist()
        
        # 确保特征目录存在
        self.cqt_dir = os.path.join(data_dir, 'cqt')
        self.timbre_dir = os.path.join(data_dir, 'timbre')
        
        if not os.path.exists(self.cqt_dir):
            raise ValueError(f"CQT特征目录不存在: {self.cqt_dir}")
        if not os.path.exists(self.timbre_dir):
            raise ValueError(f"Timbre特征目录不存在: {self.timbre_dir}")
        
        print(f"已加载数据集，共{len(self.file_ids)}个样本")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        """
        获取数据样本
        Args:
            idx: 索引
        Returns:
            cqt_feature: CQT特征 (C, H, W)
            timbre_feature: 音色特征 (D,)
            label: 标签 (1,)
        """
        file_id = self.file_ids[idx]
        
        # 加载CQT特征
        cqt_path = os.path.join(self.cqt_dir, f"{file_id}.npy")
        if not os.path.exists(cqt_path):
            raise FileNotFoundError(f"CQT特征文件不存在: {cqt_path}")
        cqt_feature = np.load(cqt_path)
        
        # 确保CQT特征形状正确 (添加通道维度)
        if len(cqt_feature.shape) == 2:
            cqt_feature = np.expand_dims(cqt_feature, axis=0)
        
        # 加载音色特征
        timbre_path = os.path.join(self.timbre_dir, f"{file_id}.npy")
        if not os.path.exists(timbre_path):
            raise FileNotFoundError(f"Timbre特征文件不存在: {timbre_path}")
        timbre_feature = np.load(timbre_path)
        
        # 获取标签
        label = self.labels_df.loc[self.labels_df['file_id'] == file_id, 'label'].values[0]
        
        # 转换为张量
        cqt_tensor = torch.FloatTensor(cqt_feature)
        timbre_tensor = torch.FloatTensor(timbre_feature)
        label_tensor = torch.FloatTensor([label])
        
        return cqt_tensor, timbre_tensor, label_tensor 