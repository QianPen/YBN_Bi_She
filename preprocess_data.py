from utils.feature_extraction import compute_cqt_stats, extract_cqt_with_normalization, extract_timbre_features
from config import Config
import numpy as np
import librosa
import os
from tqdm import tqdm
import shutil

def split_dataset(files, train_ratio=0.8):
    """将数据集分割为训练集和验证集"""
    np.random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]

def main():
    # 创建输出目录
    for dataset_type in ['train', 'val']:
        os.makedirs(f"{Config.DATA_PATH}/{dataset_type}/cqt", exist_ok=True)
        os.makedirs(f"{Config.DATA_PATH}/{dataset_type}/timbre", exist_ok=True)
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(f"{Config.DATA_PATH}/raw_audio") if f.endswith('.wav')]
    train_files, val_files = split_dataset(audio_files)
    
    # 创建标签文件
    for dataset_type, files in [('train', train_files), ('val', val_files)]:
        with open(f"{Config.DATA_PATH}/{dataset_type}_labels.csv", 'w') as f:
            f.write("file_id,label\n")
            for file in files:
                # 假设文件名格式为 NAME_X.wav，其中X是标签
                label = file.split('_')[1].split('.')[0]
                f.write(f"{file[:-4]},{label}\n")
    
    # 计算CQT统计量（仅使用训练集）
    print(f"开始处理训练集音频文件...")
    stats = compute_cqt_stats(f"{Config.DATA_PATH}/raw_audio")
    np.savez(f"{Config.DATA_PATH}/cqt_stats.npz", mean=stats['mean'], std=stats['std'])
    print("CQT统计量计算完成")
    
    # 处理训练集
    print("处理训练集...")
    for wav_file in tqdm(train_files):
        input_path = os.path.join(Config.DATA_PATH, 'raw_audio', wav_file)
        try:
            cqt = extract_cqt_with_normalization(input_path)
            output_path = f"{Config.DATA_PATH}/train/cqt/{wav_file.replace('.wav', '')}.npy"
            np.save(output_path, cqt)
            timbre = extract_timbre_features(input_path)
            np.save(f"{Config.DATA_PATH}/train/timbre/{wav_file[:-4]}.npy", timbre)
        except Exception as e:
            print(f"处理训练集文件 {wav_file} 失败: {str(e)}")
    
    # 处理验证集
    print("处理验证集...")
    for wav_file in tqdm(val_files):
        input_path = os.path.join(Config.DATA_PATH, 'raw_audio', wav_file)
        try:
            cqt = extract_cqt_with_normalization(input_path)
            output_path = f"{Config.DATA_PATH}/val/cqt/{wav_file.replace('.wav', '')}.npy"
            np.save(output_path, cqt)
            timbre = extract_timbre_features(input_path)
            np.save(f"{Config.DATA_PATH}/val/timbre/{wav_file[:-4]}.npy", timbre)
        except Exception as e:
            print(f"处理验证集文件 {wav_file} 失败: {str(e)}")
    
    print(f"处理完成！训练集: {len(train_files)}个文件，验证集: {len(val_files)}个文件")

def test_single_file():
    test_wav = "data/raw_audio/test.wav"
    cqt = extract_cqt_with_normalization(test_wav)
    print(f"测试文件特征形状：{cqt.shape}")
    np.save("data/cqt_features/test.npy", cqt)

if __name__ == "__main__":
    main()  # 先注释掉test_single_file()，用这个测试 
    # test_single_file()  # 先注释掉main()，用这个测试 
