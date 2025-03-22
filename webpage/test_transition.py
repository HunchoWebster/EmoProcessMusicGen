import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webpage.musicgen_service import create_service, MusicGenConfig, EMOTION_PRESET_PARAMS
import time

def test_transition_music():
    """测试情绪过渡音乐生成功能"""
    print("开始测试情绪过渡音乐生成功能")
    
    # 确保情绪预设包含必要的标签
    print(f"可用情绪标签: {list(EMOTION_PRESET_PARAMS.keys())}")
    
    # 检查是否包含测试所需的情绪标签
    required_emotions = ['anxiety', 'relaxed']
    missing_emotions = [e for e in required_emotions if e not in EMOTION_PRESET_PARAMS]
    if missing_emotions:
        print(f"缺少必要的情绪标签: {missing_emotions}")
        # 如果缺少必要的情绪标签，添加测试用的预设
        if 'relaxed' in missing_emotions:
            EMOTION_PRESET_PARAMS['relaxed'] = {
                "model_params": {
                    "temperature": 0.8,
                    "top_k": 200,
                    "cfg_coef": 3.0
                },
                "audio_params": {
                    "low_shelf_gain": 0.0,
                    "mid_gain": 0.0,
                    "high_shelf_gain": 0.0,
                    "target_db": -20.0,
                    "limiter_threshold": -1.0
                }
            }
            print(f"已添加 'relaxed' 情绪预设")
    
    try:
        # 创建服务实例
        print("正在创建服务实例...")
        service = create_service(
            model_name='facebook/musicgen-small',
            duration=10,
            use_sampling=True,
            top_k=250,
            temperature=1.0,
            cfg_coef=3.0,
            output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music')
        )
        
        if not service:
            print("服务创建失败")
            return False
            
        print(f"服务创建成功，使用模型: {service.config.model_name}")
        
        # 检查服务是否包含必要的方法
        if not hasattr(service, 'generate_transition_music'):
            print("服务不支持情绪过渡音乐生成功能")
            return False
            
        # 准备测试参数
        from_emotion = 'anxiety'
        to_emotion = 'relaxed'
        from_prompt = "Fast-paced, chaotic music with irregular rhythms and dissonant tones, creating a sense of tension and unease."
        to_prompt = "Calm and peaceful music with gentle melodies, soft harmonies, and a slow tempo around 60 BPM."
        transition_duration = 20
        
        # 生成过渡音乐
        print(f"开始生成情绪过渡音乐: {from_emotion} -> {to_emotion}...")
        start_time = time.time()
        success, audio_path, raw_audio_path = service.generate_transition_music(
            from_emotion=from_emotion,
            to_emotion=to_emotion,
            from_prompt=from_prompt,
            to_prompt=to_prompt,
            transition_duration=transition_duration
        )
        elapsed_time = time.time() - start_time
        print(f"过渡音乐生成耗时: {elapsed_time:.2f}秒")
        
        print(f"过渡音乐生成结果: success={success}, audio_path={audio_path}, raw_path={raw_audio_path}")
        
        if not success:
            print("过渡音乐生成失败")
            return False
            
        # 检查文件是否存在
        if not os.path.exists(audio_path) or not os.path.exists(raw_audio_path):
            print(f"音频文件生成失败，文件不存在: audio_path={os.path.exists(audio_path)}, raw_path={os.path.exists(raw_audio_path)}")
            return False
            
        print(f"过渡音乐生成成功，文件保存在: {audio_path}")
        return True
        
    except Exception as e:
        import traceback
        print(f"测试过程中发生错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    result = test_transition_music()
    print(f"测试结果: {'成功' if result else '失败'}") 