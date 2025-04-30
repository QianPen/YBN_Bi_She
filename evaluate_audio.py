import os
import torch
import numpy as np
import librosa
import argparse
from config import Config
from models.tg_critic_model import TGCritic
from utils.feature_extraction import extract_cqt_with_normalization, extract_timbre_features

def load_model(model_path='saved_models/best_model.pth'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path)
    model = TGCritic()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    
    print(f"模型已加载: {model_path}")
    return model

def evaluate_single_audio(audio_path, model):
    """评估单个音频文件"""
    try:
        # 检查音频格式
        if not audio_path.endswith('.wav'):
            print("警告：建议使用.wav格式的音频文件")
        
        # 提取特征
        print("正在提取CQT特征...")
        cqt = extract_cqt_with_normalization(audio_path)
        cqt = torch.from_numpy(cqt).unsqueeze(0).unsqueeze(0).cuda()  # 添加批次和通道维度
        
        print("正在提取音色特征...")
        timbre = extract_timbre_features(audio_path)
        timbre = torch.from_numpy(timbre).unsqueeze(0).cuda()  # 添加批次维度
        
        # 进行预测
        with torch.no_grad():
            outputs = model(cqt, timbre)
            
            if Config.TASK_TYPE == 'regression':
                score = outputs.squeeze().item()
                print(f"预测得分: {score:.4f}")
                return score
            else:  # classification
                pred = torch.argmax(outputs, dim=1).item()
                print(f"预测类别: {pred}")
                return pred
                
    except Exception as e:
        print(f"评估失败: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='评估单个音频文件')
    parser.add_argument('--audio-path', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model-path', type=str, default='saved_models/best_model.pth', help='模型文件路径')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model_path)
    
    # 评估音频
    result = evaluate_single_audio(args.audio_path, model)
    
    if result is not None:
        print("评估完成！")
    else:
        print("评估失败！")

if __name__ == '__main__':
    main() 