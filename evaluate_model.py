import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader

# 导入配置
from config import Config

# 导入自定义模块
from models.tg_critic_model import TGCritic
from dataset.dataset import VocalQualityDataset

def load_model(model_path='saved_models/best_model.pth'):
    """
    加载训练好的模型
    """
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

def save_model(model, path='saved_models', name='tg_critic_model.pth'):
    """
    保存训练好的模型
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    model_path = os.path.join(path, name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(Config.__dict__.items())  # 将mappingproxy转换为普通字典
    }, model_path)
    
    print(f"模型已保存到 {model_path}")
    return model_path

def evaluate_model(model, test_loader):
    """
    评估模型性能
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for cqt, timbre, labels in test_loader:
            # 将数据移动到GPU
            cqt = cqt.cuda().to(torch.float32)
            timbre = timbre.cuda().to(torch.float32)
            labels = labels.cuda().to(torch.float32)
            
            # 进行预测
            outputs = model(cqt, timbre)
            
            # 根据任务类型处理预测结果
            if Config.TASK_TYPE == 'regression':
                preds = (outputs.squeeze() > 0.5).float()
            elif Config.TASK_TYPE == 'binary':
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            else:  # 分类
                preds = torch.argmax(outputs, dim=1)
            
            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 打印评估结果
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 返回评估结果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

def plot_confusion_matrix(cm, save_path='figures/confusion_matrix.png'):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"混淆矩阵已保存到 {save_path}")

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='TG-CRITIC模型评估')
    parser.add_argument('--model-path', type=str, default='saved_models/best_model.pth', help='模型路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--save-model', action='store_true', help='保存评估后的模型')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model_path)
    
    # 获取测试数据加载器
    test_loader = get_dataloader(args.batch_size, 'test')
    
    # 评估模型
    results = evaluate_model(model, test_loader)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'])
    
    # 保存模型
    if args.save_model:
        save_model(model)
    
    print("评估完成!")

def get_dataloader(batch_size, dataset_type='test'):
    """
    获取数据加载器
    """
    # 数据集路径
    data_path = os.path.join(Config.DATA_PATH, dataset_type)
    
    # 检查数据集路径
    if not os.path.exists(data_path):
        raise ValueError(f"数据集路径 {data_path} 不存在。请确保数据已准备好。")
    
    # 检查是否有标签文件
    label_file = os.path.join(Config.DATA_PATH, f"{dataset_type}_labels.csv")
    if not os.path.exists(label_file):
        raise ValueError(f"标签文件 {label_file} 不存在。请先准备标签文件。")
    
    # 使用DataLoader加载数据
    dataset = VocalQualityDataset(data_path, label_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"已加载{dataset_type}数据集，共{len(dataset)}个样本")
    return dataloader

if __name__ == "__main__":
    import argparse
    import os
    
    # 确保输出目录存在
    os.makedirs('figures', exist_ok=True)
    
    main() 