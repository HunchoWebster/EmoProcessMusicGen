import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import time
import os
from openai import OpenAI
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging
import subprocess
from datetime import datetime
import warnings

# 忽略特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# 忽略 xformers 相关的警告
os.environ['XFORMERS_MORE_DETAILS'] = '0'

@dataclass
class MusicGenConfig:
    """音乐生成配置类"""
    model_name: str
    use_diffusion: bool = False
    duration: int = 10
    top_k: int = 250
    use_sampling: bool = True
    sample_rate: int = 32000
    output_dir: str = './generated_music'

    def __post_init__(self):
        """确保duration参数被正确设置"""
        if not hasattr(self, 'duration') or self.duration is None:
            self.duration = 10

class MusicGenService:
    """音乐生成服务类"""
    
    def __init__(self, api_key: Optional[str] = None, config: MusicGenConfig = None, output_dir: str = './generated_music', progress_callback=None):
        """
        初始化音乐生成服务
        
        Args:
            api_key: 豆包API密钥（可选）
            config: MusicGenConfig配置对象
            output_dir: 输出目录
            progress_callback: 进度回调函数
        """
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.model = None
        self.mbd = None
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        # 设置更高的日志级别以抑制警告
        logging.basicConfig(
            level=logging.ERROR,  # 将日志级别改为 ERROR
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 抑制其他库的日志
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('audiocraft').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        
    def initialize(self) -> bool:
        """
        初始化模型和必要组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("正在初始化模型...")
            self.model = MusicGen.get_pretrained(self.config.model_name)
            
            if self.config.use_diffusion:
                self.mbd = MultiBandDiffusion.get_mbd_musicgen()
            
            self.model.set_generation_params(
                use_sampling=self.config.use_sampling,
                top_k=self.config.top_k,
                duration=self.config.duration
            )
            
            # 确保输出目录存在
            Path(self.output_dir).mkdir(exist_ok=True)
            
            self.logger.info("模型初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            return False
    
    def create_prompt(self, emotion_text: str) -> Optional[str]:
        """
        根据情绪描述生成音乐提示词
        
        Args:
            emotion_text: 情绪描述文本
            
        Returns:
            Optional[str]: 生成的提示词，失败则返回None
        """
        if not self.api_key:
            self.logger.error("未设置API密钥")
            return None
            
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )
            
            completion = client.chat.completions.create(
                model="ep-20250220214537-p7622",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a music prompt generator for a music generation model called 'musicgen'. "
                            "Your task is to analyze the user's input, which may express a specific emotional state, "
                            "and then generate a concise, clear, detailed, and creative English music prompt that "
                            "incorporates elements to regulate or transform the stated emotion. For example, if the input "
                            "indicates sadness, include musical elements that transition from melancholy to hope. If the input "
                            "indicates anxiety, include calming, soothing elements to ease tension. Keep the output as short and clear as possible, "
                            "without any extra commentary."
                        )
                    },
                    {"role": "user", "content": emotion_text}
                ]
            )
            
            prompt = completion.choices[0].message.content.strip()
            self.logger.info(f"生成提示词: {prompt}")
            return prompt
            
        except Exception as e:
            self.logger.error(f"提示词生成失败: {str(e)}")
            return None
    
    def generate_music(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """
        生成音乐
        
        Args:
            prompt: 音乐生成提示词
            
        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 生成的音频文件路径)
        """
        if not self.model:
            self.logger.error("模型未初始化")
            return False, None
            
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"generated_music_{timestamp}.wav")
            
            # 使用原始的 MusicGen 模型生成音乐
            output = self.model.generate(
                descriptions=[prompt],
                progress=True,  # 禁用进度显示
                return_tokens=True
            )
            
            # 处理音频数据
            audio_data = output[0].cpu().numpy()
            
            if audio_data.ndim == 3:
                audio_data = audio_data.squeeze(0)
            
            if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            
            # 振幅归一化
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 保存音频文件
            sf.write(
                str(output_file),
                audio_data,
                samplerate=self.config.sample_rate
            )
            
            self.logger.info(f"音乐生成成功，保存至: {output_file}")
            
            # 处理扩散解码器输出
            if self.config.use_diffusion and self.mbd:
                diffusion_path = os.path.join(self.output_dir, f"generated_music_diffusion_{timestamp}.wav")
                out_diffusion = self.mbd.tokens_to_wav(output[1])
                sf.write(
                    str(diffusion_path),
                    out_diffusion.cpu().numpy(),
                    samplerate=self.config.sample_rate
                )
                self.logger.info(f"扩散解码器输出保存至: {diffusion_path}")
            
            return True, output_file

        except Exception as e:
            self.logger.error(f"音乐生成失败: {str(e)}")
            return False, None
            
    def process_emotion(self, emotion_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        处理情绪输入并生成音乐的完整流程
        
        Args:
            emotion_text: 情绪描述文本
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (是否成功, 生成的提示词, 音频文件路径)
        """
        # 生成提示词
        prompt = self.create_prompt(emotion_text)
        if not prompt:
            return False, None, None
            
        # 生成音乐
        success, audio_path = self.generate_music(prompt)
        return success, prompt, audio_path

def create_service(api_key: Optional[str] = None, model_name: str = 'facebook/musicgen-small', duration: int = 10, use_sampling: bool = True, top_k: int = 250, output_dir: str = './generated_music', progress_callback=None) -> MusicGenService:
    """
    创建并初始化音乐生成服务的便捷函数
    
    Args:
        api_key: 豆包API密钥（可选）
        model_name: 模型名称
        duration: 生成音乐的时长（秒）
        use_sampling: 是否使用采样
        top_k: top-k采样参数
        output_dir: 输出目录
        progress_callback: 进度回调函数
        
    Returns:
        MusicGenService: 初始化好的服务实例
    """
    config = MusicGenConfig(
        model_name=model_name,
        duration=duration,
        use_sampling=use_sampling,
        top_k=top_k,
        output_dir=output_dir
    )
    service = MusicGenService(api_key, config, output_dir, progress_callback)
    service.initialize()
    return service 