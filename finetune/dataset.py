import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Optional, Tuple

class MusicGenDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        sample_rate: int = 32000,
        duration: float = 30.0,
        transform: Optional[torch.nn.Module] = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 音频文件目录
            metadata_file: 包含音频描述和标签的JSON文件
            sample_rate: 目标采样率
            duration: 音频片段长度（秒）
            transform: 可选的音频转换
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        
        # 加载元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # 计算每个样本的采样点数
        self.samples_per_audio = int(sample_rate * duration)
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            audio: 音频张量 [channels, samples]
            text: 音频描述文本
        """
        item = self.metadata[idx]
        audio_path = os.path.join(self.data_dir, item['audio_file'])
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # 确保音频长度一致
        if waveform.shape[1] > self.samples_per_audio:
            # 随机选择一段
            start = torch.randint(0, waveform.shape[1] - self.samples_per_audio, (1,))
            waveform = waveform[:, start:start + self.samples_per_audio]
        elif waveform.shape[1] < self.samples_per_audio:
            # 填充
            pad_length = self.samples_per_audio - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
        # 应用转换
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, item['description']
    
    def get_metadata(self) -> list:
        """获取所有元数据"""
        return self.metadata 