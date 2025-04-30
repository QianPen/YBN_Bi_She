import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from config import Config

def convert_audio(input_path, output_dir):
    """将任意音频转换为标准格式"""
    try:
        # 加载并重采样
        y, sr = librosa.load(input_path, 
                            sr=Config.AUDIO_FORMAT['sample_rate'], 
                            mono=True)
        
        # 调整时长
        target_samples = int(Config.AUDIO_FORMAT['duration'] * Config.AUDIO_FORMAT['sample_rate'])
        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            y = np.pad(y, (0, max(0, target_samples - len(y))), mode='constant')
        
        # 保存文件
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        sf.write(output_path, y,
                samplerate=Config.AUDIO_FORMAT['sample_rate'],
                subtype=Config.AUDIO_FORMAT['subtype'],
                format=Config.AUDIO_FORMAT['format'])
        return output_path
    except librosa.util.exceptions.ParameterError as pe:
        print(f"参数错误：{input_path} - {str(pe)}")
    except Exception as e:
        print(f"未知错误：{input_path} - {str(e)}")
    return None

def batch_convert(input_dir, output_dir):
    """批量转换目录下所有音频文件"""
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a')
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    # 使用进度条显示转换进度
    for fname in tqdm(audio_files, desc="Converting files"):
        input_path = os.path.join(input_dir, fname)
        convert_audio(input_path, output_dir) 