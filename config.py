import os

class Config:
    # ================== 数据配置 ==================
    DATA_PATH = './data'  # 数据根目录
    SR = 16000            # 音频采样率
    FRAME_LENGTH = 256    # 每帧样本数
    
    # ================== 特征配置 ==================
    CQT_BINS = 96         # CQT频带数
    HOP_LENGTH = 512      # CQT帧移
    TIMBRE_DIM = 128      # 音色特征维度
    
    # ================== 模型架构配置 ==================
    MODEL_ARCH = 'fusion'  # 可选: 
                           # 'cqt_only' - 仅使用CQT特征
                           # 'timbre_only' - 仅使用音色特征
                           # 'fusion' - 特征融合（默认）
    
    # 任务类型配置
    TASK_TYPE = 'classification'  # 可选: 'regression', 'classification'
    
    # ================== 训练超参数 ==================
    BATCH_SIZE = 9       # 批大小（根据显存调整）
    LR = 3e-5             # 初始学习率（范围：1e-6 ~ 1e-4）
    EPOCHS = 100          # 最大训练轮次
    DROPOUT_RATE = 0.3    # Dropout概率
    GRAD_CLIP = 5.0       # 梯度裁剪
    EARLY_STOP = True     # 是否启用早停
    
    # ================== 优化器配置 ==================
    OPTIMIZER = 'adamw'   # 可选: 'adam', 'sgd', 'rmsprop'
    WEIGHT_DECAY = 1e-4   # 权重衰减系数
    MOMENTUM = 0.9        # SGD动量参数
    
    # ================== 学习率调度 ==================
    SCHEDULER = 'plateau' # 可选: 
                          # 'plateau' - 根据验证损失调整
                          # 'cosine' - 余弦退火
                          # 'step' - 阶梯下降
    LR_FACTOR = 0.5       # 学习率衰减因子
    LR_PATIENCE = 5       # 学习率调整耐心值
    
    # ================== 早停机制 ==================
    PATIENCE = 10         # 早停等待轮次
    DELTA = 0.001         # 视为改进的最小变化量
    
    # ================== 硬件配置 ==================
    AMP_ENABLED = True    # 启用混合精度训练
    NUM_WORKERS = 4       # 数据加载线程数
    
    # ================== 路径配置 ==================
    MODEL_SAVE_PATH = './checkpoints'  # 模型保存路径
    LOG_DIR = './logs'    # 训练日志目录
    
    # ================== 模型结构参数 ==================
    # CQT分支参数
    CQT_CHANNELS = [64, 128, 256]  # 卷积通道数
    CQT_KERNELS = [3, 3, 3]       # 卷积核尺寸
    
    # 音色分支参数
    TIMBRE_LAYERS = [256, 128]    # 全连接层维度
    
    # 融合层参数
    FUSION_LAYERS = [512, 256]    # 融合后全连接层
    
    # 新增内容验证参数
    REQUIRE_VOCAL_ONLY = True  # 必须为纯人声
    ALLOW_BACKGROUND = False   # 是否允许背景音 
    
    # 最低质量标准
    MIN_SNR = 20  # 信噪比(dB)
    MAX_CLIPPING = 0.5  # 最大削波比例 
    
    # 音频格式规范
    AUDIO_FORMAT = {
        'format': 'wav',
        'subtype': 'PCM_16',
        'channels': 1,
        'sample_rate': 16000,
        'duration': 8.0  # 固定8秒长度
    }
    
    # 新增音色模型路径
    TIMBRE_MODEL_PATH = './models/timbre_model.pth'
    
    # 新增混合精度配置
    AMP_DTYPE = 'bfloat16'  # 可选: 'float16'/'bfloat16'
    SCALER_GROWTH_INTERVAL = 2000  # 动态缩放间隔
    
    # 新增梯度控制参数
    GRAD_CLIP_TYPE = 'norm'  # 可选 'value'/'norm'
    GRAD_CLIP_VAL = 1.0      # 梯度裁剪阈值
    GRAD_ACCUM_STEPS = 2     # 梯度累积步数
    