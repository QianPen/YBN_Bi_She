from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from config import Config
import os
import torch

class VocalDataset(Dataset):
    def __init__(self, cqt_dir, timbre_dir, label_file, stats_file):
        self.cqt_features = []
        # 按样本维度堆叠
        for fname in os.listdir(cqt_dir):
            if fname.endswith('.npy'):
                cqt = np.load(os.path.join(cqt_dir, fname)).real  # 只保留实部
                self.cqt_features.append(cqt.T)  # 转置为(256, 96)
        self.cqt_features = np.stack(self.cqt_features)  # 形状变为(N, 256, 96)
        
        self.timbre_features = np.load(f"{timbre_dir}/labels.npy", allow_pickle=True)
        self.labels = np.load(label_file)
        
        # 标准化处理
        stats = np.load(stats_file)
        self.cqt_features = (self.cqt_features - stats['mean'].reshape(1, 1, -1)) / (stats['std'].reshape(1, 1, -1) + 1e-6)

        # 检查数据形状
        print(f"Shape of cqt_features: {self.cqt_features.shape}")
        print(f"Shape of timbre_features: {self.timbre_features.shape}")
        print(f"Shape of labels: {self.labels.shape}")

        # 加载原始标签后添加处理
        if Config.TASK_TYPE == 'regression':
            self.labels = self.labels.astype(np.float32)
        elif Config.TASK_TYPE in ['classification', 'binary']:
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cqt = torch.FloatTensor(self.cqt_features[idx])
        timbre = torch.FloatTensor([self.timbre_features[idx]])
        label = torch.FloatTensor([self.labels[idx]])
        
        return (cqt, timbre), label



def get_dataloader(batch_size=32):
    dataset = VocalDataset(
        cqt_dir=os.path.join(Config.DATA_PATH, 'cqt_features'),
        timbre_dir=os.path.join(Config.DATA_PATH, 'timbre_embeddings'),
        label_file=os.path.join(Config.DATA_PATH, 'labels', 'labels.npy'),
        stats_file=os.path.join(Config.DATA_PATH, 'cqt_stats.npz')
    )
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"First cqt shape: {dataset[0][0].shape}")
    print(f"First timbre shape: {dataset[0][1].shape}")
    print(f"First label: {dataset[0][2]}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def validate_audio_purity(audio_path):
    """简单验证是否为人声主导"""
    y, sr = librosa.load(audio_path)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(spectral_centroid) > 2000  # 人声频率范围阈值
