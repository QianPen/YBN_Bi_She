import os
import numpy as np
from config import Config

def auto_generate_labels():
    """根据音频文件数量生成伪标签"""
    audio_files = os.listdir(f"{Config.DATA_PATH}/raw_audio")
    num_samples = len([f for f in audio_files if f.endswith('.wav')])
    
    # 生成随机标签（仅供测试，实际应使用真实标签）
    labels = np.random.randint(0, 3, size=num_samples)
    np.save(f"{Config.DATA_PATH}/labels/labels.npy", labels)
    print(f"已生成{num_samples}个样本的标签文件")

if __name__ == "__main__":
    auto_generate_labels() 