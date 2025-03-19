import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import time
import os
import json
import re
from openai import OpenAI
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging
import subprocess
from datetime import datetime
import warnings
from .audio_processor import AudioProcessor

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
        self.audio_processor = AudioProcessor(sample_rate=self.config.sample_rate)
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        # 将日志级别改为 INFO 以显示详细信息
        logging.basicConfig(
            level=logging.INFO,  # 从 ERROR 改为 INFO
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 修改其他库的日志级别，但保持主要日志为 INFO
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
        Generate music prompt based on emotional description
        
        Args:
            emotion_text: Emotional description text
            
        Returns:
            Optional[str]: Generated prompt, returns None if failed
        """
        if not self.api_key:
            self.logger.error("API key not set")
            return None
            
        try:
            self.logger.info(f"开始调用 API，输入文本: {emotion_text}")
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )
            
            # 记录完整的请求消息
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional music therapy prompt generator. Your task is to analyze the user's emotional description "
                        "and generate two parts:\n"
                        "1. Emotion Label: Identify the primary emotion type from the user's description\n"
                        "2. Music Prompt: Generate a detailed English music generation prompt\n\n"
                        "You must respond with ONLY a valid JSON object in the following format:\n"
                        "{\n"
                        '  "emotion_label": "emotion label",\n'
                        '  "music_prompt": "music generation prompt"\n'
                        "}\n\n"
                        "The emotion label must be exactly one of: anxiety, depression, insomnia, stress, distraction, pain, trauma\n"
                        "The music prompt must:\n"
                        "- Be in English\n"
                        "- Include specific musical elements (rhythm, timbre, melodic features, etc.)\n"
                        "- Target emotional regulation goals\n"
                        "- Be concise and clear\n"
                        "- Not exceed 150 characters\n\n"
                        "Example valid response:\n"
                        "{\n"
                        '  "emotion_label": "anxiety",\n'
                        '  "music_prompt": "Calm ambient music with gentle piano melodies, soft strings, and slow tempo around 60 BPM"\n'
                        "}"
                    )
                },
                {"role": "user", "content": emotion_text}
            ]
            
            self.logger.info("API 请求消息:")
            self.logger.info(f"System: {messages[0]['content']}")
            self.logger.info(f"User: {messages[1]['content']}")
            
            completion = client.chat.completions.create(
                model="ep-20250220214537-p7622",
                messages=messages
            )
            
            # 记录完整的 API 响应
            self.logger.info("API 完整响应:")
            self.logger.info(f"Model: {completion.model}")
            self.logger.info(f"Created: {completion.created}")
            self.logger.info(f"Usage: {completion.usage}")
            self.logger.info(f"Choices: {len(completion.choices)}")
            
            for i, choice in enumerate(completion.choices):
                self.logger.info(f"Choice {i}:")
                self.logger.info(f"  Index: {choice.index}")
                self.logger.info(f"  Message Role: {choice.message.role}")
                self.logger.info(f"  Message Content: {choice.message.content}")
                self.logger.info(f"  Finish Reason: {choice.finish_reason}")
            
            # 记录原始响应内容，包括所有可能的格式
            response = completion.choices[0].message.content.strip()
            self.logger.info("原始响应内容:")
            self.logger.info("=" * 50)
            self.logger.info(response)
            self.logger.info("=" * 50)
            
            # 尝试不同的响应格式
            self.logger.info("尝试解析响应...")
            
            # 1. 尝试直接解析为 JSON
            try:
                result = json.loads(response)
                self.logger.info("成功直接解析为 JSON")
            except json.JSONDecodeError:
                self.logger.info("直接解析 JSON 失败，尝试其他方法")
                
                # 2. 尝试查找 JSON 对象
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
                    self.logger.info("成功提取 JSON 对象")
                    try:
                        result = json.loads(response)
                        self.logger.info("成功解析提取的 JSON 对象")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"解析提取的 JSON 对象失败: {str(e)}")
                        return None
                else:
                    self.logger.warning("未找到 JSON 对象")
                    return None
            
            # 记录解析后的结果
            self.logger.info("解析结果:")
            self.logger.info(f"  Raw Result: {result}")
            
            # Validate required fields
            if 'emotion_label' not in result or 'music_prompt' not in result:
                self.logger.error("Missing required fields in JSON response")
                self.logger.error(f"Available fields: {list(result.keys())}")
                return None
            
            # Validate emotion label
            valid_emotions = {'anxiety', 'depression', 'insomnia', 'stress', 'distraction', 'pain', 'trauma'}
            if result['emotion_label'] not in valid_emotions:
                self.logger.error(f"Invalid emotion label: {result['emotion_label']}")
                self.logger.error(f"Valid emotions: {valid_emotions}")
                return None
            
            # Validate music prompt
            if not isinstance(result['music_prompt'], str):
                self.logger.error("Music prompt must be a string")
                self.logger.error(f"Actual type: {type(result['music_prompt'])}")
                return None
            
            prompt_length = len(result['music_prompt'])
            if prompt_length > 150:
                self.logger.warning(f"Music prompt is too long ({prompt_length} chars), truncating to 150 chars")
                result['music_prompt'] = result['music_prompt'][:150]
            
            self.logger.info("最终结果:")
            self.logger.info(f"  Emotion Label: {result['emotion_label']}")
            self.logger.info(f"  Music Prompt: {result['music_prompt']}")
            self.logger.info(f"  Prompt Length: {len(result['music_prompt'])} chars")
            return result['music_prompt']
            
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {str(e)}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def generate_music(self, prompt: str, eq_params: Optional[Dict[str, float]] = None, target_db: float = -20.0, limiter_threshold: float = -1.0) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        生成音乐
        
        Args:
            prompt: 音乐生成提示词
            eq_params: 均衡器参数字典，包含 low_shelf_gain, mid_gain, high_shelf_gain
            target_db: 目标响度值（dB）
            limiter_threshold: 限幅阈值（dB）
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (是否成功, 生成的音频文件路径, 原始音频文件路径)
        """
        if not self.model:
            self.logger.error("模型未初始化")
            return False, None, None
            
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"generated_music_{timestamp}.wav")
            raw_output_file = os.path.join(self.output_dir, f"raw_music_{timestamp}.wav")
            
            # 使用原始的 MusicGen 模型生成音乐
            output = self.model.generate(
                descriptions=[prompt],
                progress=True,
                return_tokens=True
            )
            
            # 处理音频数据
            audio_data = output[0].cpu().numpy()
            
            if audio_data.ndim == 3:
                audio_data = audio_data.squeeze(0)
            
            if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            
            # 保存原始音频
            sf.write(
                str(raw_output_file),
                audio_data,
                samplerate=self.config.sample_rate
            )
            
            # 应用音频处理
            processed_audio = self.audio_processor.process_audio(
                audio_data,
                normalize=True,
                target_db=target_db,
                eq_params=eq_params
            )
            
            # 应用限幅器
            processed_audio = self.audio_processor.apply_limiter(processed_audio, limiter_threshold)
            
            # 保存处理后的音频文件
            sf.write(
                str(output_file),
                processed_audio,
                samplerate=self.config.sample_rate
            )
            
            self.logger.info(f"音乐生成成功，保存至: {output_file}")
            
            # 处理扩散解码器输出
            if self.config.use_diffusion and self.mbd:
                diffusion_path = os.path.join(self.output_dir, f"generated_music_diffusion_{timestamp}.wav")
                raw_diffusion_path = os.path.join(self.output_dir, f"raw_music_diffusion_{timestamp}.wav")
                out_diffusion = self.mbd.tokens_to_wav(output[1])
                # 保存原始扩散解码器输出
                out_diffusion_np = out_diffusion.cpu().numpy()
                sf.write(
                    str(raw_diffusion_path),
                    out_diffusion_np,
                    samplerate=self.config.sample_rate
                )
                # 对扩散解码器输出也应用相同的音频处理
                out_diffusion = self.audio_processor.process_audio(
                    out_diffusion_np,
                    normalize=True,
                    target_db=-14.0,
                    eq_params=eq_params
                )
                sf.write(
                    str(diffusion_path),
                    out_diffusion,
                    samplerate=self.config.sample_rate
                )
                self.logger.info(f"扩散解码器输出保存至: {diffusion_path}")
            
            return True, output_file, raw_output_file

        except Exception as e:
            self.logger.error(f"音乐生成失败: {str(e)}")
            return False, None, None
            
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
        success, audio_path, raw_audio_path = self.generate_music(prompt)
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