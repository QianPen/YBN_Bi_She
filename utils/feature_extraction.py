import librosa
import numpy as np
import glob
from tqdm import tqdm
from config import Config
import torch
import torch.nn as nn

def extract_cqt(audio_path, sr=16000, hop_length=512, n_bins=96):
    print(f"正在处理：{audio_path}")  # 添加调试信息
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        print(f"加载成功，音频长度：{len(y)/sr:.2f}秒")
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length,
                        n_bins=n_bins, bins_per_octave=24,
                        fmin=librosa.note_to_hz('E2'))
        print(f"CQT形状：{cqt.shape}")
        return np.pad(cqt, ((0,0), (0, max(0, 256 - cqt.shape[1]))), mode='constant')[:, :256]
    except Exception as e:
        print(f"处理异常：{str(e)}")
        raise

def extract_timbre_features(audio_path, model_path=None):
    """提取音色特征，不依赖预训练模型"""
    try:
        # 加载音频
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 提取MFCC特征作为音色特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 提取其他音色相关特征
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 计算统计特征
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        centroid_mean = np.mean(spectral_centroid)
        rolloff_mean = np.mean(spectral_rolloff)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # 组合所有特征
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [centroid_mean],
            [rolloff_mean],
            contrast_mean
        ])
        
        # 确保特征维度为128
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
        else:
            features = features[:128]
            
        return features
    except Exception as e:
        print(f"提取音色特征失败: {str(e)}")
        # 返回随机特征作为后备方案
        return np.random.rand(128)

def split_and_pad(audio, sr, target_length=8):
    """将音频分割并填充为固定长度"""
    target_samples = target_length * sr
    # 确保长度是hop_length的整数倍
    target_samples = (target_samples // Config.HOP_LENGTH) * Config.HOP_LENGTH
    if len(audio) >= target_samples:
        return audio[:target_samples]
    else:
        return np.pad(audio, (0, target_samples - len(audio)), mode='constant')

def extract_cqt_with_normalization(audio_path, sr=16000, stats=None):
    # 重采样和长度处理
    y, _ = librosa.load(audio_path, sr=sr)
    y = split_and_pad(y, sr)
    
    # 提取CQT
    cqt = librosa.cqt(y, sr=sr, 
                     hop_length=Config.HOP_LENGTH,
                     n_bins=Config.CQT_BINS,
                     bins_per_octave=24,
                     fmin=librosa.note_to_hz('E2'))
    
    # 标准化处理
    if stats is not None:
        cqt = (cqt - stats['mean']) / (stats['std'] + 1e-6)
    return cqt[:, :Config.FRAME_LENGTH]

def compute_cqt_stats(dataset_path):
    """计算训练集的CQT均值和方差"""
    all_cqts = []
    for audio_path in tqdm(glob.glob(f"{dataset_path}/*.wav")):
        try:
            cqt = extract_cqt(audio_path)
            all_cqts.append(cqt)
        except Exception as e:
            print(f"处理 {audio_path} 失败: {str(e)}")
    
    if not all_cqts:
        raise ValueError("没有找到有效的音频文件")
    
    all_cqts = np.concatenate(all_cqts, axis=1)
    return {
        'mean': np.mean(all_cqts, axis=1, keepdims=True),
        'std': np.std(all_cqts, axis=1, keepdims=True)
    } 