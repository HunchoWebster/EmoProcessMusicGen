import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import logging

# 忽略警告
warnings.filterwarnings('ignore')
# 设置日志级别
logging.getLogger().setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, send_file, Response
from webpage.musicgen_service import create_service, MusicGenConfig
from pathlib import Path
import json
import re

# 创建Flask应用，指定静态文件目录
app = Flask(__name__, static_folder='static', static_url_path='/static')
# 设置 Flask 的日志级别
app.logger.setLevel(logging.ERROR)

os.environ["ARK_API_KEY"] = "9e5d6644-81ee-4360-baf1-db1816b2c344"

# 添加一个全局变量来存储生成进度
generation_progress = {
    'prompt_status': 0,    # 提示词生成进度
    'music_status': 0      # 音乐生成进度
}

# 创建音乐生成服务实例
service = create_service(
    api_key=os.getenv("ARK_API_KEY"),
    model_name='facebook/musicgen-small',
    duration=10,  # 确保时长为10秒
    use_sampling=True,
    top_k=250,
    output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music'),  # 使用绝对路径
    progress_callback=None  # 移除进度回调
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress', methods=['GET'])
def get_progress():
    """获取生成进度"""
    return jsonify(generation_progress)

@app.route('/generate', methods=['POST'])
def generate_music():
    try:
        global generation_progress
        # 重置进度
        generation_progress = {
            'prompt_status': 0,
            'music_status': 0
        }
        
        print("收到生成请求")
        emotion_text = request.json.get('emotion_text')
        print(f"情绪文本: {emotion_text}")
        
        if not emotion_text:
            return jsonify({'success': False, 'error': '请输入情绪描述'})

        # 更新提示词生成进度
        generation_progress['prompt_status'] = 50
        print("开始生成提示词")
        prompt = service.create_prompt(emotion_text)
        generation_progress['prompt_status'] = 100
        
        print(f"生成的提示词: {prompt}")
        
        if not prompt:
            return jsonify({'success': False, 'error': '提示词生成失败'})

        # 先返回提示词
        return jsonify({
            'success': True,
            'prompt': prompt,
            'stage': 'prompt'  # 添加阶段标识
        })
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_music', methods=['POST'])
def generate_music_audio():
    try:
        global generation_progress
        prompt = request.json.get('prompt')
        
        if not prompt:
            generation_progress['music_status'] = 0  # 重置进度
            return jsonify({'success': False, 'error': '提示词不能为空'})

        # 重置音乐生成进度
        generation_progress['music_status'] = 0
        print("开始生成音乐")
        success, audio_path = service.generate_music(prompt)
        
        if not success:
            generation_progress['music_status'] = 0  # 重置进度
            return jsonify({'success': False, 'error': '音乐生成失败'})
            
        generation_progress['music_status'] = 100  # 只在成功时设置100%
        print(f"音乐生成结果: success={success}, path={audio_path}")
            
        audio_filename = os.path.basename(audio_path)
        relative_path = f'generated_music/{audio_filename}'
        
        return jsonify({
            'success': True,
            'audio_path': relative_path,
            'stage': 'music'  # 添加阶段标识
        })
        
    except Exception as e:
        generation_progress['music_status'] = 0  # 重置进度
        print(f"发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # 确保静态文件目录存在
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    generated_music_dir = os.path.join(static_dir, 'generated_music')
    Path(generated_music_dir).mkdir(parents=True, exist_ok=True)
    app.run(debug=True) 