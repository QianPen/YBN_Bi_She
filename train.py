import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

# 导入配置
from config import Config

# 导入自定义模块
from models.tg_critic_model import TGCritic
from dataset.dataset import VocalQualityDataset

# 忽略警告
warnings.filterwarnings("ignore")

def get_dataloader(batch_size, dataset_type='train'):
    """
    获取数据加载器
    Args:
        batch_size: 批大小
        dataset_type: 'train' 或 'val'
    Returns:
        DataLoader对象
    """
    # 数据集路径
    data_path = os.path.join(Config.DATA_PATH, dataset_type)
    
    # 检查数据集路径
    if not os.path.exists(data_path):
        raise ValueError(f"数据集路径 {data_path} 不存在")
    
    # 检查特征目录
    cqt_dir = os.path.join(data_path, 'cqt')
    timbre_dir = os.path.join(data_path, 'timbre')
    if not os.path.exists(cqt_dir) or not os.path.exists(timbre_dir):
        raise ValueError(f"特征目录不存在: {cqt_dir} 或 {timbre_dir}")
    
    # 检查标签文件
    label_file = os.path.join(Config.DATA_PATH, f"{dataset_type}_labels.csv")
    if not os.path.exists(label_file):
        raise ValueError(f"标签文件不存在: {label_file}")
    
    # 读取标签文件
    labels_df = pd.read_csv(label_file)
    if len(labels_df) == 0:
        raise ValueError(f"标签文件为空: {label_file}")
    
    # 验证标签值
    if Config.TASK_TYPE == 'binary':
        if not all(labels_df['label'].isin([0, 1])):
            raise ValueError("二分类任务的标签必须是0或1")
    elif Config.TASK_TYPE == 'classification':
        if not all(labels_df['label'].apply(lambda x: isinstance(x, int) and x >= 0)):
            raise ValueError("分类任务的标签必须是非负整数")
    elif Config.TASK_TYPE == 'regression':
        if not all(labels_df['label'].apply(lambda x: isinstance(x, (int, float)))):
            raise ValueError("回归任务的标签必须是数值类型")
    
    # 调整批大小
    if len(labels_df) < batch_size:
        print(f"警告: {dataset_type}数据集样本数({len(labels_df)})小于批大小({batch_size})，将使用较小的批大小")
        batch_size = min(batch_size, len(labels_df))
    
    # 使用DataLoader加载数据
    dataset = VocalQualityDataset(data_path, label_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if dataset_type == 'train' else False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"已加载{dataset_type}数据集，共{len(dataset)}个样本")
    return dataloader

def train_1s_strategy(model, train_loader, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for cqt, timbre, labels in train_loader:
            outputs = model(cqt, timbre)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_2s_strategy(model, train_loader, epochs_stage1=50, epochs_stage2=50):
    # 第一阶段：训练高分辨率分支
    for param in model.timbre_branch.parameters():
        param.requires_grad = False
    
    # 训练hr_branch...
    
    # 第二阶段：冻结hr_branch，训练timbre_branch
    for param in model.hr_branch.parameters():
        param.requires_grad = False
    for param in model.timbre_branch.parameters():
        param.requires_grad = True
    
    # 训练timbre_branch... 

def train_with_amp(model, train_loader, val_loader, epochs):
    # 创建保存模型的目录
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    if Config.TASK_TYPE == 'regression':
        criterion = nn.MSELoss()
    else:  # classification
        criterion = nn.CrossEntropyLoss()
    
    # Initialize the early stopping counter
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    # 用于跟踪评估指标
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    
    # 更新为新版API - 注意这里使用torch.amp而不是torch.cuda.amp
    scaler = torch.amp.GradScaler(enabled=Config.AMP_ENABLED)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, (cqt, timbre, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 检查批次大小
            if cqt.size(0) == 1:
                print(f"警告: 批次 {batch_idx} 只包含一个样本，可能会影响批归一化层")
            
            # 将数据转移到GPU并确保数据类型一致
            cqt = cqt.cuda().to(torch.float32)
            timbre = timbre.cuda().to(torch.float32)
            labels = labels.cuda().to(torch.float32)
            
            # 更新为新版API - 使用'cuda'设备类型
            with torch.amp.autocast(device_type='cuda', enabled=Config.AMP_ENABLED):
                outputs = model(cqt, timbre)
                
                # 根据任务类型处理标签和损失
                if Config.TASK_TYPE == 'regression':
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    preds = (outputs.squeeze() > 0.5).float()  # 阈值处理
                else:  # classification
                    loss = criterion(outputs, labels.squeeze().long())
                    preds = torch.argmax(outputs, dim=1)
                
                # 收集预测和标签用于计算指标
                all_train_preds.extend(preds.detach().cpu().numpy())
                all_train_labels.extend(labels.squeeze().cpu().numpy())
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if Config.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        if len(all_train_preds) > 0:
            train_accuracy = accuracy_score(all_train_labels, all_train_preds)
            train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
            train_accuracies.append(train_accuracy)
            train_f1_scores.append(train_f1)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch_idx, (cqt, timbre, labels) in enumerate(val_loader):
                # 检查批次大小
                if cqt.size(0) == 1:
                    print(f"警告: 验证批次 {batch_idx} 只包含一个样本")
                
                # 将数据转移到GPU并确保数据类型一致
                cqt = cqt.cuda().to(torch.float32)
                timbre = timbre.cuda().to(torch.float32)
                labels = labels.cuda().to(torch.float32)
                
                # 更新为新版API
                with torch.amp.autocast(device_type='cuda', enabled=Config.AMP_ENABLED):
                    outputs = model(cqt, timbre)
                    
                    # 根据任务类型处理标签和损失
                    if Config.TASK_TYPE == 'regression':
                        loss = criterion(outputs.squeeze(), labels.squeeze())
                        preds = (outputs.squeeze() > 0.5).float()
                    else:  # classification
                        loss = criterion(outputs, labels.squeeze().long())
                        preds = torch.argmax(outputs, dim=1)
                    
                    # 收集预测和标签
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.squeeze().cpu().numpy())
                
                val_loss += loss.item()
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        if len(all_val_preds) > 0:
            val_accuracy = accuracy_score(all_val_labels, all_val_preds)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            
            print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        else:
            print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 只保存必要的配置信息
            config_dict = {
                'DATA_PATH': Config.DATA_PATH,
                'SR': Config.SR,
                'FRAME_LENGTH': Config.FRAME_LENGTH,
                'CQT_BINS': Config.CQT_BINS,
                'HOP_LENGTH': Config.HOP_LENGTH,
                'TIMBRE_DIM': Config.TIMBRE_DIM,
                'MODEL_ARCH': Config.MODEL_ARCH,
                'TASK_TYPE': Config.TASK_TYPE,
                'BATCH_SIZE': Config.BATCH_SIZE,
                'LR': Config.LR,
                'EPOCHS': Config.EPOCHS,
                'DROPOUT_RATE': Config.DROPOUT_RATE,
                'GRAD_CLIP': Config.GRAD_CLIP,
                'EARLY_STOP': Config.EARLY_STOP,
                'OPTIMIZER': Config.OPTIMIZER,
                'WEIGHT_DECAY': Config.WEIGHT_DECAY,
                'MOMENTUM': Config.MOMENTUM,
                'SCHEDULER': Config.SCHEDULER,
                'LR_FACTOR': Config.LR_FACTOR,
                'LR_PATIENCE': Config.LR_PATIENCE,
                'PATIENCE': Config.PATIENCE,
                'DELTA': Config.DELTA,
                'AMP_ENABLED': Config.AMP_ENABLED,
                'NUM_WORKERS': Config.NUM_WORKERS,
                'MODEL_SAVE_PATH': Config.MODEL_SAVE_PATH,
                'LOG_DIR': Config.LOG_DIR
            }
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config_dict
            }, 'saved_models/best_model.pth')
            print(f"模型已保存 (Val Loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config_dict
            }, f'saved_models/checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        if Config.EARLY_STOP and early_stop_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # 训练结束后保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1],
        'train_loss': train_losses[-1],
        'config': {
            'DATA_PATH': Config.DATA_PATH,
            'SR': Config.SR,
            'FRAME_LENGTH': Config.FRAME_LENGTH,
            'CQT_BINS': Config.CQT_BINS,
            'HOP_LENGTH': Config.HOP_LENGTH,
            'TIMBRE_DIM': Config.TIMBRE_DIM,
            'MODEL_ARCH': Config.MODEL_ARCH,
            'TASK_TYPE': Config.TASK_TYPE,
            'BATCH_SIZE': Config.BATCH_SIZE,
            'LR': Config.LR,
            'EPOCHS': Config.EPOCHS,
            'DROPOUT_RATE': Config.DROPOUT_RATE,
            'GRAD_CLIP': Config.GRAD_CLIP,
            'EARLY_STOP': Config.EARLY_STOP,
            'OPTIMIZER': Config.OPTIMIZER,
            'WEIGHT_DECAY': Config.WEIGHT_DECAY,
            'MOMENTUM': Config.MOMENTUM,
            'SCHEDULER': Config.SCHEDULER,
            'LR_FACTOR': Config.LR_FACTOR,
            'LR_PATIENCE': Config.LR_PATIENCE,
            'PATIENCE': Config.PATIENCE,
            'DELTA': Config.DELTA,
            'AMP_ENABLED': Config.AMP_ENABLED,
            'NUM_WORKERS': Config.NUM_WORKERS,
            'MODEL_SAVE_PATH': Config.MODEL_SAVE_PATH,
            'LOG_DIR': Config.LOG_DIR
        }
    }, 'saved_models/final_model.pth')
    print("训练完成，最终模型已保存")
    
    # 返回训练历史
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores
    }

def check_audio_format(filepath):
    """验证音频格式是否符合要求"""
    try:
        y, sr = librosa.load(filepath, sr=None)
        assert sr == Config.SR, f"采样率应为{Config.SR}，实际为{sr}"
        assert y.ndim == 1, "应使用单声道音频"
        return True
    except Exception as e:
        print(f"文件{filepath}格式错误：{str(e)}")
        return False 

def load_audio(filepath):
    """自动处理不同格式的音频文件"""
    try:
        return librosa.load(filepath, sr=Config.SR, mono=True)
    except:
        # 使用ffmpeg作为后备解码器
        os.system(f"ffmpeg -i {filepath} -ar {Config.SR} -ac 1 temp.wav")
        y, _ = librosa.load("temp.wav", sr=Config.SR)
        os.remove("temp.wav")
        return y 

def compute_loss(outputs, labels):
    criterion = nn.MSELoss()  # 使用均方误差损失
    loss = criterion(outputs, labels.float())  # 将 labels 转换为 float 类型
    return loss

def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 10))
    
    # 绘制损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(history['train_losses'], label='训练损失')
    plt.plot(history['val_losses'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(3, 1, 2)
    plt.plot(history['train_accuracies'], label='训练准确率')
    plt.plot(history['val_accuracies'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1分数曲线
    plt.subplot(3, 1, 3)
    plt.plot(history['train_f1_scores'], label='训练F1分数')
    plt.plot(history['val_f1_scores'], label='验证F1分数')
    plt.title('F1分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/training_history.png')
    print("训练历史曲线已保存至 figures/training_history.png")
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='TG-CRITIC训练脚本')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE, help='批大小')
    parser.add_argument('--lr', type=float, default=Config.LR, help='学习率')
    parser.add_argument('--dropout', type=float, default=Config.DROPOUT_RATE, help='Dropout比例')
    parser.add_argument('--grad-clip', type=float, default=Config.GRAD_CLIP, help='梯度裁剪值')
    parser.add_argument('--early-stop', action='store_true', help='启用早停')
    parser.add_argument('--tb-log', action='store_true', help='启用TensorBoard日志')
    parser.add_argument('--comment', type=str, default='', help='实验备注')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='训练轮数')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式（只训练2轮）')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.DROPOUT_RATE = args.dropout
    Config.GRAD_CLIP = args.grad_clip
    Config.EARLY_STOP = args.early_stop
    Config.AMP_ENABLED = args.amp
    Config.EPOCHS = 2 if args.quick_test else args.epochs  # 快速测试模式只跑2轮
    
    # 检查批大小是否合理
    if Config.BATCH_SIZE < 2:
        print("警告: 批大小小于2可能导致BatchNorm层错误。自动调整批大小为2。")
        Config.BATCH_SIZE = 2
    
    # 打印训练配置
    print("="*50)
    print(f"训练配置:")
    print(f"批大小: {Config.BATCH_SIZE}")
    print(f"学习率: {Config.LR}")
    print(f"Dropout: {Config.DROPOUT_RATE}")
    print(f"梯度裁剪: {Config.GRAD_CLIP}")
    print(f"训练轮数: {Config.EPOCHS}" + (" (快速测试模式)" if args.quick_test else ""))
    print(f"早停: {'启用' if Config.EARLY_STOP else '禁用'}")
    print(f"混合精度: {'启用' if Config.AMP_ENABLED else '禁用'}")
    print("="*50)
    
    # 确保目录存在
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 初始化模型
    model = TGCritic().cuda()
    
    # 获取数据加载器
    train_loader = get_dataloader(Config.BATCH_SIZE, 'train')  
    val_loader = get_dataloader(Config.BATCH_SIZE, 'val')  # 验证集
    
    # 启用TensorBoard日志
    if args.tb_log:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.comment}"))
            # 记录模型图
            sample_cqt = torch.randn(2, 1, Config.CQT_BINS, 100).cuda()  # 使用批大小为2
            sample_timbre = torch.randn(2, Config.TIMBRE_DIM).cuda()
            writer.add_graph(model, (sample_cqt, sample_timbre))
        except Exception as e:
            print(f"无法初始化TensorBoard: {str(e)}")
            writer = None
    else:
        writer = None
    
    # 开始训练
    history = train_with_amp(model, train_loader, val_loader, Config.EPOCHS)
    
    # 绘制训练曲线
    plot_training_history(history)
    
    print("训练已完成！请使用evaluate_model.py评估模型性能。")
    
def create_dummy_labels():
    """
    创建示例标签文件（如果不存在）
    """
    for dataset_type in ['train', 'val', 'test']:
        label_file = os.path.join(Config.DATA_PATH, f"{dataset_type}_labels.csv")
        if not os.path.exists(label_file):
            print(f"创建示例标签文件: {label_file}")
            
            # 创建包含多个样本的标签文件（至少30个样本，确保每批次有足够数据）
            file_ids = [f'sample_{i:03d}' for i in range(1, 31)]
            labels = [i % 2 for i in range(1, 31)]  # 交替的0和1标签
            
            df = pd.DataFrame({
                'file_id': file_ids,
                'label': labels
            })
            df.to_csv(label_file, index=False)
            
            # 为这些样本创建示例特征文件
            for file_id in file_ids:
                cqt_path = os.path.join(Config.DATA_PATH, dataset_type, 'cqt', f"{file_id}.npy")
                timbre_path = os.path.join(Config.DATA_PATH, dataset_type, 'timbre', f"{file_id}.npy")
                
                if not os.path.exists(cqt_path):
                    # 创建随机CQT特征 (1, 96, 100) - CQT是二维特征
                    np.save(cqt_path, np.random.rand(1, Config.CQT_BINS, 100))
                
                if not os.path.exists(timbre_path):
                    # 创建随机音色特征 (128,)
                    np.save(timbre_path, np.random.rand(Config.TIMBRE_DIM))

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # 确保数据目录结构存在
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_PATH, 'train', 'cqt'), exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_PATH, 'train', 'timbre'), exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_PATH, 'val', 'cqt'), exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_PATH, 'val', 'timbre'), exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_PATH, 'test', 'cqt'), exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_PATH, 'test', 'timbre'), exist_ok=True)
    
    # 创建示例标签文件（如果不存在）
    create_dummy_labels()
    
    main()

