import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple, Optional
import logging
import librosa
import time
from datetime import datetime

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
            
    def create_transition(self, from_audio: np.ndarray, to_audio: np.ndarray, overlap_seconds: float = 2.0) -> np.ndarray:
        """
        创建两段音频之间的平滑过渡
        
        Args:
            from_audio: 起始音频数组
            to_audio: 目标音频数组
            overlap_seconds: 重叠部分的秒数
            
        Returns:
            过渡后的音频数组
        """
        overall_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.logger.info(f"[{timestamp}] ===== 开始创建音频过渡 =====")
        
        try:
            # 记录输入参数
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 起始音频形状: {from_audio.shape}, 类型: {from_audio.dtype}")
            self.logger.info(f"[{timestamp}] 目标音频形状: {to_audio.shape}, 类型: {to_audio.dtype}")
            self.logger.info(f"[{timestamp}] 重叠时长: {overlap_seconds} 秒")
            
            # 确保音频具有正确的维度 (通道数, 样本数)
            if from_audio.ndim == 1:
                from_audio = from_audio.reshape(1, -1)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 转换起始音频为2D: {from_audio.shape}")
            if to_audio.ndim == 1:
                to_audio = to_audio.reshape(1, -1)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 转换目标音频为2D: {to_audio.shape}")
                
            # 确保两个音频有相同的通道数
            if from_audio.shape[0] != to_audio.shape[0]:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.warning(f"[{timestamp}] 音频通道数不匹配: {from_audio.shape[0]} vs {to_audio.shape[0]}")
                
                # 取最小通道数
                channels = min(from_audio.shape[0], to_audio.shape[0])
                from_audio = from_audio[:channels]
                to_audio = to_audio[:channels]
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 已调整为相同通道数: {channels}，新形状 - 起始: {from_audio.shape}, 目标: {to_audio.shape}")
                
            # 计算重叠样本数
            overlap_samples = int(overlap_seconds * self.sample_rate)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 重叠样本数: {overlap_samples}")
            
            # 确保重叠样本数不超过音频长度
            from_length = from_audio.shape[1]
            to_length = to_audio.shape[1]
            
            if overlap_samples >= from_length or overlap_samples >= to_length:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.warning(f"[{timestamp}] 重叠样本数过大(原始值:{overlap_samples})，调整为更小的值")
                overlap_samples = min(from_length // 2, to_length // 2)
                self.logger.info(f"[{timestamp}] 调整后的重叠样本数: {overlap_samples}")
                
            # 创建线性淡入淡出权重
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 创建交叉淡变曲线...")
            try:
                # 线性淡变曲线
                fade_out = np.linspace(1.0, 0.0, overlap_samples)
                fade_in = np.linspace(0.0, 1.0, overlap_samples)
                
                # 检查曲线是否包含无效值
                if np.isnan(fade_in).any() or np.isnan(fade_out).any():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.logger.warning(f"[{timestamp}] 淡变曲线包含NaN值，将进行修复")
                    fade_in = np.nan_to_num(fade_in)
                    fade_out = np.nan_to_num(fade_out)
                    
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 淡变曲线长度: {len(fade_in)}")
            except Exception as curve_error:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.error(f"[{timestamp}] 创建淡变曲线失败: {str(curve_error)}")
                # 使用备选方案
                fade_out = np.ones(overlap_samples) * 0.5
                fade_in = np.ones(overlap_samples) * 0.5
                self.logger.info(f"[{timestamp}] 使用均匀0.5权重代替淡变曲线")
                
            # 截取音频片段进行混合
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 准备混合音频片段...")
            
            try:
                # 获取起始音频的末尾部分
                from_end = from_audio[:, -overlap_samples:]
                # 获取目标音频的开始部分
                to_start = to_audio[:, :overlap_samples]
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 起始音频末尾形状: {from_end.shape}, 目标音频开头形状: {to_start.shape}")
                
                # 检查音频片段是否包含无效值
                if np.isnan(from_end).any() or np.isinf(from_end).any():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.logger.warning(f"[{timestamp}] 起始音频末尾包含无效值，将进行修复")
                    from_end = np.nan_to_num(from_end)
                    
                if np.isnan(to_start).any() or np.isinf(to_start).any():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.logger.warning(f"[{timestamp}] 目标音频开头包含无效值，将进行修复")
                    to_start = np.nan_to_num(to_start)
                
                # 混合音频片段
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 开始混合过渡片段...")
                
                mixed_overlap = (from_end * fade_out.reshape(1, -1) + 
                                to_start * fade_in.reshape(1, -1))
                
                # 检查混合结果是否包含无效值
                if np.isnan(mixed_overlap).any() or np.isinf(mixed_overlap).any():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.logger.warning(f"[{timestamp}] 混合片段包含无效值，将进行修复")
                    mixed_overlap = np.nan_to_num(mixed_overlap)
                    
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 混合片段形状: {mixed_overlap.shape}")
            except Exception as mix_error:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.error(f"[{timestamp}] 混合音频片段失败: {str(mix_error)}")
                import traceback
                self.logger.error(f"[{timestamp}] 错误堆栈: {traceback.format_exc()}")
                
                # 创建一个空白的混合片段
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 使用备选方案: 创建一个静默过渡片段")
                mixed_overlap = np.zeros((from_audio.shape[0], overlap_samples))
                
            # 构建最终的过渡音频
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 构建最终的过渡音频...")
                
            try:
                # 获取不重叠的部分
                from_start = from_audio[:, :-overlap_samples]
                to_end = to_audio[:, overlap_samples:]
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 起始非重叠部分形状: {from_start.shape}, 目标非重叠部分形状: {to_end.shape}")
                
                # 拼接三个部分: 起始非重叠部分 + 混合重叠部分 + 目标非重叠部分
                result = np.concatenate([from_start, mixed_overlap, to_end], axis=1)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 结果音频形状: {result.shape}")
                
            except Exception as concat_error:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.error(f"[{timestamp}] 拼接音频失败: {str(concat_error)}")
                import traceback
                self.logger.error(f"[{timestamp}] 错误堆栈: {traceback.format_exc()}")
                
                # 简单拼接，不做混合
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 使用备选方案: 简单拼接音频")
                result = np.concatenate([from_audio, to_audio], axis=1)
                self.logger.info(f"[{timestamp}] 简单拼接结果形状: {result.shape}")
            
            # 检查结果音频质量
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 检查结果音频...")
            
            # 检查是否有NaN或Inf
            if np.isnan(result).any() or np.isinf(result).any():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.warning(f"[{timestamp}] 结果音频包含NaN或Inf值，将修复")
                result = np.nan_to_num(result)
                
            # 检查是否有爆音
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.warning(f"[{timestamp}] 检测到可能的爆音，最大值: {max_val}，将进行归一化")
                result = result / max_val * 0.95
                
            # 计算信噪比（简单估计）
            try:
                signal_power = np.mean(result**2)
                noise_floor = 1e-10  # 避免除以零
                snr_db = 10 * np.log10(signal_power / noise_floor) if signal_power > noise_floor else 0
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 估计信噪比: {snr_db:.2f} dB")
            except Exception as snr_error:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.warning(f"[{timestamp}] 计算信噪比失败: {str(snr_error)}")
            
            overall_end_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(f"[{timestamp}] 音频过渡创建成功，总耗时: {overall_end_time - overall_start_time:.2f}秒")
            self.logger.info(f"[{timestamp}] ===== 音频过渡创建完成 =====")
            
            return result
            
        except Exception as e:
            overall_end_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.error(f"[{timestamp}] 创建音频过渡失败: {str(e)}")
            import traceback
            self.logger.error(f"[{timestamp}] 错误堆栈: {traceback.format_exc()}")
            self.logger.error(f"[{timestamp}] 总耗时(失败): {overall_end_time - overall_start_time:.2f}秒")
            self.logger.error(f"[{timestamp}] ===== 音频过渡创建失败 =====")
            
            # 尝试备选方案：简单连接两个音频
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 尝试备选方案: 简单拼接两个音频")
                
                # 确保音频格式一致
                if from_audio.ndim == 1:
                    from_audio = from_audio.reshape(1, -1)
                if to_audio.ndim == 1:
                    to_audio = to_audio.reshape(1, -1)
                
                # 保证通道数匹配
                if from_audio.shape[0] != to_audio.shape[0]:
                    channels = min(from_audio.shape[0], to_audio.shape[0])
                    from_audio = from_audio[:channels]
                    to_audio = to_audio[:channels]
                
                # 修复任何NaN或Inf
                from_audio = np.nan_to_num(from_audio)
                to_audio = np.nan_to_num(to_audio)
                
                # 简单连接
                result = np.concatenate([from_audio, to_audio], axis=1)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.info(f"[{timestamp}] 备选方案成功，结果形状: {result.shape}")
                return result
            except Exception as fallback_error:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.logger.error(f"[{timestamp}] 备选方案也失败: {str(fallback_error)}")
                
                # 最终备选：返回第一个音频
                try:
                    if isinstance(from_audio, np.ndarray) and from_audio.size > 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.logger.info(f"[{timestamp}] 使用最终备选方案: 仅返回起始音频")
                        return from_audio
                    elif isinstance(to_audio, np.ndarray) and to_audio.size > 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.logger.info(f"[{timestamp}] 使用最终备选方案: 仅返回目标音频")
                        return to_audio
                    else:
                        # 创建一个空白音频
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.logger.warning(f"[{timestamp}] 无法使用任何输入音频，生成空白音频")
                        return np.zeros((1, int(self.sample_rate * 5)))  # 5秒空白
                except Exception as final_error:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.logger.error(f"[{timestamp}] 所有备选方案都失败: {str(final_error)}")
                    # 返回一个最小的可用音频
                    return np.zeros((1, int(self.sample_rate * 1)))  # 1秒空白
        
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

    def apply_equalizer(self, audio_data, params):
        """应用均衡器"""
        try:
            # 确保音频数据长度足够
            if len(audio_data) <= 15:
                self.logger.warning("音频数据长度不足，跳过均衡器处理")
                return audio_data
                
            # 应用均衡器
            low_shelf = self.apply_low_shelf(audio_data, params.get('low_shelf_gain', 0))
            mid = self.apply_mid(audio_data, params.get('mid_gain', 0))
            high_shelf = self.apply_high_shelf(audio_data, params.get('high_shelf_gain', 0))
            
            # 混合所有频段
            return low_shelf + mid + high_shelf
            
        except Exception as e:
            self.logger.error(f"均衡器处理失败: {str(e)}")
            return audio_data  # 发生错误时返回原始音频 