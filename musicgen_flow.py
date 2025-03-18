import numpy as np
import torch
import torchaudio
import soundfile as sf
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from pathlib import Path
import time
from openai import OpenAI
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
from audiocraft.utils.notebook import display_audio

import os



def init_model():
    """
    初始化MusicGen模型和相关参数
    """ 
    # API密钥设置
    os.environ["ARK_API_KEY"] = "9e5d6644-81ee-4360-baf1-db1816b2c344"
    api_key = os.environ.get("ARK_API_KEY")
    print("环境变量中的API密钥:", api_key)
    
    pause = input("按回车继续...")
    
    # 初始化模型
    use_diffusion = False
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    mbd = MultiBandDiffusion.get_mbd_musicgen() if use_diffusion else None

    # 设置生成参数
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=10
    )
    
    return {
        'model': model,
        'api_key': api_key,
        'use_diffusion': use_diffusion,
        'mbd': mbd
    }



def create_music_prompt(user_input, api_key):
    """
    接收用户输入，调用API生成仅包含音乐提示词的简短英文prompt。
    """
    if api_key is None:
        print("请设置api_key环境变量以继续。")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    try:
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
                {"role": "user", "content": user_input}
            ])
        final_prompt = completion.choices[0].message.content.strip()
        return final_prompt
    except Exception as e:
        print(f"Music prompt generation error: {e}")
        return "a gentle ambient piece with soft pads that gradually builds a sense of tranquility"

def generate_music(model, prompt, use_diffusion=False, mbd=None):
    """
    生成音乐的主函数，接收已生成的prompt和必要的模型参数
    """
    try:
        # 生成音乐
        output = model.generate(
            descriptions=[prompt],
            progress=True,
            return_tokens=True
        )
        
        # 处理音频数据
        audio_data = output[0].cpu().numpy()
        
        # 维度处理
        if audio_data.ndim == 3:
            audio_data = audio_data.squeeze(0)
        
        if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
        
        # 振幅归一化
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 生成时间戳
        timestamp = int(time.time())
        
        # 创建输出目录
        output_dir = Path("./generated_music")
        output_dir.mkdir(exist_ok=True)
        
        # 保存音频文件
        audio_path = output_dir / f"generated_music_{timestamp}.wav"
        sf.write(str(audio_path), audio_data, samplerate=32000)
        
        print(f"音乐生成成功！文件保存在: {audio_path}")
        
        # 修改扩散解码器部分
        if use_diffusion and mbd is not None:
            out_diffusion = mbd.tokens_to_wav(output[1])
            diffusion_path = output_dir / f"generated_music_diffusion_{timestamp}.wav"
            sf.write(str(diffusion_path), 
                    out_diffusion.cpu().numpy(),
                    samplerate=32000)
            print(f"扩散解码器输出保存在: {diffusion_path}")
            
        return True
        
    except Exception as e:
        print(f"生成过程中出错: {str(e)}")
        return False

def main():
    print("欢迎使用情绪调节音乐生成器！")
    
    # 初始化并获取所有必要参数
    print("初始化模型...")
    config = init_model()
    
    print("请输入您当前的情绪描述，例如：'今天工作压力很大，需要放松'")
    print("输入 'q' 退出程序")
    
    while True:
        user_input = input("\n请输入您的情绪描述: ")
        if user_input.lower() == 'q':
            break
            
        print("\n正在生成提示词...")
        # 生成提示词
        prompt = create_music_prompt(user_input, config['api_key'])
        if prompt is None:
            print("提示词生成失败，请重试。")
            continue
            
        print(f"生成的提示词: {prompt}")
        print("\n正在生成音乐，请稍候...")
        
        # 使用生成的提示词生成音乐，传入必要参数
        success = generate_music(
            model=config['model'],
            prompt=prompt,
            use_diffusion=config['use_diffusion'],
            mbd=config['mbd']
        )
        
        if success:
            print("\n音乐生成完成！")
        else:
            print("\n音乐生成失败，请重试。")

if __name__ == "__main__":
    main() 