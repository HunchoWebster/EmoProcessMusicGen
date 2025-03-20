import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple, Optional
import logging
import librosa

class AudioProcessor:
    """音频处理类，提供音量归一化、均衡器和限幅器功能"""
    
    def __init__(self, sample_rate: int = 32000):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率，默认32000Hz
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
    def normalize_volume(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        对音频进行音量归一化处理
        
        Args:
            audio: 输入音频数据
            target_db: 目标响度值（dB），默认-20 LUFS
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            # 计算当前音频的RMS值
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms == 0:
                return audio
                
            # 计算目标RMS值
            target_rms = 10 ** (target_db / 20)
            
            # 计算增益系数
            gain = target_rms / current_rms
            
            # 应用增益
            normalized_audio = audio * gain
            
            return normalized_audio
            
        except Exception as e:
            self.logger.error(f"音量归一化处理失败: {str(e)}")
            return audio
            
    def apply_eq(self, audio: np.ndarray, 
                 low_shelf_gain: float = 0.0,
                 mid_gain: float = 0.0,
                 high_shelf_gain: float = 0.0,
                 low_freq: float = 100.0,
                 mid_low_freq: float = 1000.0,
                 mid_high_freq: float = 5000.0,
                 high_freq: float = 8000.0) -> np.ndarray:
        """
        应用均衡器处理
        
        Args:
            audio: 输入音频数据
            low_shelf_gain: 低频增益（dB），默认0
            mid_gain: 中频增益（dB），默认0
            high_shelf_gain: 高频增益（dB），默认0
            low_freq: 低频截止频率（Hz），默认100
            mid_low_freq: 中频下限频率（Hz），默认1000
            mid_high_freq: 中频上限频率（Hz），默认5000
            high_freq: 高频截止频率（Hz），默认8000
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            processed_audio = audio.copy()
            
            # 设计低架滤波器
            if low_shelf_gain != 0:
                b, a = signal.butter(2, low_freq/(self.sample_rate/2), btype='low')
                low_freq = signal.filtfilt(b, a, processed_audio)
                processed_audio = processed_audio + (low_freq * (10 ** (low_shelf_gain / 20) - 1))
                
            # 设计中频滤波器
            if mid_gain != 0:
                b, a = signal.butter(2, [mid_low_freq/(self.sample_rate/2), mid_high_freq/(self.sample_rate/2)], btype='band')
                mid_freq = signal.filtfilt(b, a, processed_audio)
                processed_audio = processed_audio + (mid_freq * (10 ** (mid_gain / 20) - 1))
                
            # 设计高架滤波器
            if high_shelf_gain != 0:
                b, a = signal.butter(2, high_freq/(self.sample_rate/2), btype='high')
                high_freq = signal.filtfilt(b, a, processed_audio)
                processed_audio = processed_audio + (high_freq * (10 ** (high_shelf_gain / 20) - 1))
                
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"均衡器处理失败: {str(e)}")
            return audio
            
    def apply_limiter(self, audio: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
        """
        应用限幅器处理
        
        Args:
            audio: 输入音频数据
            threshold_db: 限幅阈值（dB），默认-1.0
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            threshold = 10 ** (threshold_db / 20)
            return np.clip(audio, -threshold, threshold)
        except Exception as e:
            self.logger.error(f"限幅器处理失败: {str(e)}")
            return audio
            
    def process_audio(self, audio: np.ndarray, 
                     normalize: bool = True,
                     target_db: float = -20.0,
                     eq_params: Optional[dict] = None) -> np.ndarray:
        """
        完整的音频处理流程
        
        Args:
            audio: 输入音频数据
            normalize: 是否进行音量归一化
            target_db: 目标响度值（dB）
            eq_params: 均衡器参数字典
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            processed_audio = audio.copy()
            
            # 应用音量归一化
            if normalize:
                processed_audio = self.normalize_volume(processed_audio, target_db)
                
            # 应用均衡器
            if eq_params:
                processed_audio = self.apply_eq(
                    processed_audio,
                    low_shelf_gain=eq_params.get('low_shelf_gain', 0.0),
                    mid_gain=eq_params.get('mid_gain', 0.0),
                    high_shelf_gain=eq_params.get('high_shelf_gain', 0.0),
                    low_freq=eq_params.get('low_freq', 100.0),
                    mid_low_freq=eq_params.get('mid_low_freq', 1000.0),
                    mid_high_freq=eq_params.get('mid_high_freq', 5000.0),
                    high_freq=eq_params.get('high_freq', 8000.0)
                )
                
            # 应用限幅器
            processed_audio = self.apply_limiter(processed_audio)
                
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"音频处理失败: {str(e)}")
            return audio
            
    def create_transition(self, from_audio, to_audio, overlap_seconds=2.0):
        """
        创建两段音频之间的平滑过渡
        
        Args:
            from_audio: 起始音频数据
            to_audio: 目标音频数据
            overlap_seconds: 重叠部分的长度（秒）
            
        Returns:
            numpy.ndarray: 过渡后的音频数据
        """
        # 确保音频是2D的
        if from_audio.ndim == 1:
            from_audio = from_audio.reshape(1, -1)
        if to_audio.ndim == 1:
            to_audio = to_audio.reshape(1, -1)
            
        # 确保两段音频的通道数相同
        assert from_audio.shape[0] == to_audio.shape[0], "两段音频的通道数必须相同"
        
        # 计算重叠部分的长度（样本数）
        overlap_samples = int(overlap_seconds * self.sample_rate)
        
        # 创建重叠部分的淡入淡出窗口
        fade_in = np.linspace(0, 1, overlap_samples)
        fade_out = np.linspace(1, 0, overlap_samples)
        
        # 计算最终音频的长度
        total_length = from_audio.shape[1] + to_audio.shape[1] - overlap_samples
        result = np.zeros((from_audio.shape[0], total_length))
        
        # 复制第一段音频（除了重叠部分用淡出处理）
        result[:, :from_audio.shape[1]] = from_audio
        for i in range(from_audio.shape[0]):
            result[i, from_audio.shape[1]-overlap_samples:from_audio.shape[1]] *= fade_out
        
        # 复制第二段音频（开始部分用淡入处理）
        offset = from_audio.shape[1] - overlap_samples
        result[:, offset:offset+to_audio.shape[1]] += to_audio
        for i in range(to_audio.shape[0]):
            result[i, offset:offset+overlap_samples] *= fade_in
        
        return result
        
    def create_smooth_transition(self, from_audio, to_audio, overlap_seconds=2.0):
        """
        创建两段音频之间的平滑过渡，使用更平滑的交叉淡变
        
        Args:
            from_audio: 起始音频数据
            to_audio: 目标音频数据
            overlap_seconds: 重叠部分的长度（秒）
            
        Returns:
            numpy.ndarray: 过渡后的音频数据
        """
        # 确保音频是2D的
        if from_audio.ndim == 1:
            from_audio = from_audio.reshape(1, -1)
        if to_audio.ndim == 1:
            to_audio = to_audio.reshape(1, -1)
            
        # 确保两段音频的通道数相同
        assert from_audio.shape[0] == to_audio.shape[0], "两段音频的通道数必须相同"
        
        # 计算重叠部分的长度（样本数）
        overlap_samples = int(overlap_seconds * self.sample_rate)
        
        # 创建重叠部分的淡入淡出窗口 - 使用余弦窗口实现更平滑的过渡
        fade_in = np.sin(np.linspace(0, np.pi/2, overlap_samples))**2
        fade_out = np.sin(np.linspace(np.pi/2, 0, overlap_samples))**2
        
        # 计算最终音频的长度
        total_length = from_audio.shape[1] + to_audio.shape[1] - overlap_samples
        result = np.zeros((from_audio.shape[0], total_length))
        
        # 复制第一段音频（除了重叠部分用淡出处理）
        result[:, :from_audio.shape[1]] = from_audio
        for i in range(from_audio.shape[0]):
            result[i, from_audio.shape[1]-overlap_samples:from_audio.shape[1]] *= fade_out
        
        # 复制第二段音频（开始部分用淡入处理）
        offset = from_audio.shape[1] - overlap_samples
        result[:, offset:offset+to_audio.shape[1]] += to_audio
        for i in range(to_audio.shape[0]):
            result[i, offset:offset+overlap_samples] *= fade_in
        
        return result
        
    def create_spectral_transition(self, from_audio, to_audio, overlap_seconds=2.0):
        """
        创建两段音频之间的频谱过渡，使用STFT进行平滑过渡
        
        Args:
            from_audio: 起始音频数据
            to_audio: 目标音频数据
            overlap_seconds: 重叠部分的长度（秒）
            
        Returns:
            numpy.ndarray: 过渡后的音频数据
        """
        # 确保音频是单通道的（因为librosa.stft需要单通道）
        if from_audio.ndim > 1:
            from_audio = np.mean(from_audio, axis=0)
        if to_audio.ndim > 1:
            to_audio = np.mean(to_audio, axis=0)
            
        # 计算重叠部分的长度（样本数）
        overlap_samples = int(overlap_seconds * self.sample_rate)
        
        # 提取需要过渡的部分
        from_end = from_audio[-overlap_samples:]
        to_start = to_audio[:overlap_samples]
        
        # 计算短时傅里叶变换
        n_fft = 2048
        hop_length = 512
        
        from_stft = librosa.stft(from_end, n_fft=n_fft, hop_length=hop_length)
        to_stft = librosa.stft(to_start, n_fft=n_fft, hop_length=hop_length)
        
        # 创建线性过渡权重
        num_frames = min(from_stft.shape[1], to_stft.shape[1])
        weights = np.linspace(1, 0, num_frames)
        
        # 线性混合两个STFT
        transition_stft = from_stft[:, :num_frames] * weights + to_stft[:, :num_frames] * (1 - weights)
        
        # 反变换回时域
        transition_audio = librosa.istft(transition_stft, hop_length=hop_length)
        
        # 组合三段音频
        result = np.concatenate([
            from_audio[:-overlap_samples],
            transition_audio,
            to_audio[overlap_samples:]
        ])
        
        # 将结果重新变为原始格式
        if from_audio.ndim > 1 or to_audio.ndim > 1:
            result = result.reshape(1, -1)
            
        return result 