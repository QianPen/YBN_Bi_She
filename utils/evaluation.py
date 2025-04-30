def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)
    acc = (preds == labels).sum().item() / labels.size(0)
    
    # 计算皮尔逊相关系数
    score = F.softmax(outputs, dim=1)[:, 0]  # 假设第一个类别为质量评分
    pearson = np.corrcoef(score.cpu().numpy(), labels.cpu().numpy())[0,1]
    return acc, pearson 

def audio_quality_check(y, sr):
    """基础音频质量检查"""
    report = {}
    
    # 检查削波
    clipping = np.mean(np.abs(y) > 0.99)
    report['clipping'] = clipping < Config.MAX_CLIPPING
    
    # 计算信噪比
    noise_profile = y[:2000]  # 假设前2000个样本是静音段
    snr = 10*np.log10(np.mean(y**2)/np.mean(noise_profile**2))
    report['snr'] = snr > Config.MIN_SNR
    
    return all(report.values()), report 