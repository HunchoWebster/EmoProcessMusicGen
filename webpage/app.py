import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import logging
from datetime import datetime
import soundfile as sf
import time

# 忽略警告
warnings.filterwarnings('ignore')
# 设置日志级别
logging.getLogger().setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, send_file, Response, url_for
from webpage.musicgen_service import create_service, MusicGenConfig, EMOTION_PRESET_PARAMS, MusicGenService
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

# 全局服务实例和初始化标志
service = None
service_initialized = False

def initialize_service():
    """初始化服务的函数"""
    global service, service_initialized
    try:
        # 检查是否已经初始化
        if service_initialized:
            print("服务已经初始化")
            return
            
        print("开始初始化服务...")
        service = create_service_with_retry(max_retries=2)
        if service is None:
            raise Exception("无法初始化音乐生成服务")
        print(f"服务初始化完成，使用模型: {service.config.model_name}")
        service_initialized = True
            
    except Exception as e:
        print(f"服务初始化失败: {str(e)}")
        print("应用将继续启动，但音乐生成功能可能不可用")
        service = None

# 创建音乐生成服务实例
def create_service_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"尝试初始化服务 (尝试 {attempt + 1}/{max_retries})...")
            
            # 首先尝试style模型
            service = create_service(
                api_key=os.getenv("ARK_API_KEY"),
                model_name='facebook/musicgen-small',  # 使用small模型
                duration=10,  # 确保时长为10秒
                use_sampling=True,
                top_k=250,
                temperature=1.0,
                cfg_coef=3.0,
                output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music'),
                progress_callback=None
            )
            print("服务创建成功，正在初始化模型...")
            if service.model is not None:
                print(f"模型已加载: {service.model.name}")
                return service
            print(f"初始化尝试 {attempt + 1} 失败 - 模型为空")
            
            # 如果style模型失败，尝试备用模型
            if attempt == 0:  # 只在第一次失败后尝试
                try:
                    print("尝试使用备用模型 'facebook/musicgen-small'...")
                    service = create_service(
                        api_key=os.getenv("ARK_API_KEY"),
                        model_name='facebook/musicgen-small',
                        duration=10,
                        use_sampling=True,
                        top_k=250,
                        temperature=1.0,
                        cfg_coef=3.0,
                        output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music'),
                        progress_callback=None
                    )
                    if service.model is not None:
                        print(f"备用模型加载成功: {service.model.name}")
                        return service
                except Exception as backup_err:
                    print(f"备用模型加载失败: {str(backup_err)}")
                
        except Exception as e:
            print(f"初始化尝试 {attempt + 1} 出错: {str(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
        
        if attempt < max_retries - 1:
            print(f"等待10秒后重试...")
            time.sleep(10)
    
    # 创建一个最小可用版本
    try:
        print("尝试创建最小可用版本...")
        config = MusicGenConfig(
            model_name='facebook/musicgen-small',
            duration=10,
            use_sampling=True,
            top_k=250,
            temperature=1.0,
            cfg_coef=3.0,
            output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music')
        )
        service = MusicGenService(os.getenv("ARK_API_KEY"), config, 
                                  os.path.join(os.path.dirname(__file__), 'static', 'generated_music'))
        print("最小服务创建成功")
        return service
    except Exception as e:
        print(f"最小服务创建失败: {str(e)}")
        raise Exception(f"服务初始化完全失败：{str(e)}")

# 初始化服务
initialize_service()

@app.before_request
def before_request():
    """在每个请求之前检查服务是否已初始化"""
    global service, service_initialized
    if not service_initialized:
        initialize_service()

@app.route('/')
def index():
    return render_template('index.html')

<<<<<<< HEAD
@app.route('/transition')
def transition():
    return render_template('transition.html')
=======
@app.route('/test', methods=['GET'])
def test_endpoint():
    """测试端点，用于验证服务是否正常响应"""
    print("测试端点被访问")
    return jsonify({
        'success': True,
        'message': '服务正常响应',
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
>>>>>>> b9a2a286e64c2c50619d59d32c47e010cb8c64a8

@app.route('/progress', methods=['GET'])
def get_progress():
    """获取生成进度"""
    return jsonify(generation_progress)

@app.route('/emotion_presets', methods=['GET'])
def get_emotion_presets():
    """获取情绪预设参数"""
    try:
        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '服务不可用，请联系管理员'})
            
        return jsonify({
            'success': True,
            'presets': EMOTION_PRESET_PARAMS
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取情绪预设失败: {str(e)}'
        })

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

        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音乐生成服务不可用，请联系管理员'})
            
        # 更新提示词生成进度
        generation_progress['prompt_status'] = 50
        print("开始生成提示词")
        result = service.create_prompt(emotion_text)
        generation_progress['prompt_status'] = 100
        
        if not result:
            return jsonify({'success': False, 'error': '提示词生成失败'})

        print(f"生成的提示词: {result['music_prompt']}")
        print(f"情绪标签: {result['emotion_label']}")

        # 返回提示词和情绪标签
        return jsonify({
            'success': True,
            'prompt': result['music_prompt'],
            'emotion_label': result['emotion_label'],
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
        emotion_label = request.json.get('emotion_label')
        duration = request.json.get('duration', 10)  # 获取时长参数，默认10秒
        eq_params = request.json.get('eq_params', None)
        target_db = request.json.get('target_db', -20.0)
        limiter_threshold = request.json.get('limiter_threshold', -1.0)
        
        if not prompt:
            generation_progress['music_status'] = 0
            return jsonify({'success': False, 'error': '提示词不能为空'})

        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音乐生成服务不可用，请联系管理员'})

        # 更新服务配置中的duration参数
        service.config.duration = duration

        # 如果提供了情绪标签，则应用情绪预设
        if emotion_label and emotion_label in EMOTION_PRESET_PARAMS:
            service.apply_emotion_preset(emotion_label)
            
            # 如果未提供均衡器参数，则使用情绪预设的参数
            if eq_params is None:
                audio_params = service.get_audio_params_for_emotion(emotion_label)
                eq_params = {
                    'low_shelf_gain': audio_params['low_shelf_gain'],
                    'mid_gain': audio_params['mid_gain'],
                    'high_shelf_gain': audio_params['high_shelf_gain']
                }
                target_db = audio_params['target_db']
                limiter_threshold = audio_params['limiter_threshold']

        # 重置音乐生成进度
        generation_progress['music_status'] = 0
        print("开始生成音乐")
        success, audio_path, raw_audio_path = service.generate_music(
            prompt, 
            eq_params=eq_params,
            target_db=target_db,
            limiter_threshold=limiter_threshold
        )
        
        if not success:
            generation_progress['music_status'] = 0  # 重置进度
            return jsonify({'success': False, 'error': '音乐生成失败'})
            
        generation_progress['music_status'] = 100  # 只在成功时设置100%
        print(f"音乐生成结果: success={success}, path={audio_path}")
            
        audio_filename = os.path.basename(audio_path)
        raw_audio_filename = os.path.basename(raw_audio_path)
        relative_path = f'generated_music/{audio_filename}'
        raw_relative_path = f'generated_music/{raw_audio_filename}'
        
        return jsonify({
            'success': True,
            'audio_path': relative_path,
            'raw_audio_path': raw_relative_path,
            'emotion_label': emotion_label,
            'stage': 'music'  # 添加阶段标识
        })
        
    except Exception as e:
        generation_progress['music_status'] = 0  # 重置进度
        print(f"发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_transition', methods=['POST'])
def generate_transition():
    """
    生成情绪过渡音乐
    """
    import time
    import traceback
    from datetime import datetime
    
    overall_start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    app.logger.info(f"[{timestamp}] ===== 接收到情绪过渡音乐生成请求 =====")
    app.logger.info(f"[{timestamp}] 请求数据: {request.json}")
    
    try:
        # 获取参数
        data = request.json
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        app.logger.info(f"[{timestamp}] 接收到请求参数: {data}")
        
        # 检查参数是否存在
        required_fields = ['from_emotion', 'to_emotion', 'from_prompt', 'to_prompt']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 缺少必要参数: {missing_fields}")
            return jsonify({
                "error": f"缺少必要参数: {missing_fields}",
                "status": "error"
            }), 400
            
        # 获取情绪和提示词
        from_emotion = data.get('from_emotion', '').strip().lower()
        to_emotion = data.get('to_emotion', '').strip().lower()
        from_prompt = data.get('from_prompt', '').strip()
        to_prompt = data.get('to_prompt', '').strip()
        transition_duration = int(data.get('transition_duration', 30))
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        app.logger.info(f"[{timestamp}] 处理后参数: from_emotion={from_emotion}, to_emotion={to_emotion}, transition_duration={transition_duration}")
        app.logger.info(f"[{timestamp}] 起始提示词: {from_prompt}")
        app.logger.info(f"[{timestamp}] 目标提示词: {to_prompt}")
        
        # 验证情绪标签
        valid_emotions = set(EMOTION_PRESET_PARAMS.keys())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        app.logger.info(f"[{timestamp}] 有效情绪标签: {valid_emotions}")
        
        if from_emotion not in valid_emotions:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 无效的起始情绪标签: {from_emotion}")
            return jsonify({
                "error": f"无效的起始情绪标签: {from_emotion}，有效标签: {list(valid_emotions)}",
                "status": "error"
            }), 400
            
        if to_emotion not in valid_emotions:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 无效的目标情绪标签: {to_emotion}")
            return jsonify({
                "error": f"无效的目标情绪标签: {to_emotion}，有效标签: {list(valid_emotions)}",
                "status": "error"
            }), 400
            
        # 检查提示词
        if not from_prompt:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 起始提示词为空")
            return jsonify({
                "error": "起始提示词不能为空",
                "status": "error"
            }), 400
            
        if not to_prompt:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 目标提示词为空")
            return jsonify({
                "error": "目标提示词不能为空",
                "status": "error"
            }), 400
            
        # 检查时长
        if transition_duration < 10:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.warning(f"[{timestamp}] 过渡时长过短，调整为最小值: 10秒")
            transition_duration = 10
        elif transition_duration > 60:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.warning(f"[{timestamp}] 过渡时长过长，调整为最大值: 60秒")
            transition_duration = 60
            
        # 检查音乐服务是否已初始化
        if not service:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 音乐服务未初始化")
            return jsonify({
                "error": "音乐服务未初始化，请稍后再试",
                "status": "error"
            }), 500
            
        # 检查模型是否已加载
        if not service.model:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.warning(f"[{timestamp}] 模型未加载，尝试加载模型")
            try:
                load_start = time.time()
                service.load_model()
                load_end = time.time()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                app.logger.info(f"[{timestamp}] 模型加载成功，耗时: {load_end - load_start:.2f}秒")
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                app.logger.error(f"[{timestamp}] 模型加载失败: {str(e)}")
                app.logger.error(f"[{timestamp}] 错误堆栈: {traceback.format_exc()}")
                return jsonify({
                    "error": f"模型加载失败: {str(e)}",
                    "status": "error"
                }), 500
                
        # 生成过渡音乐
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        app.logger.info(f"[{timestamp}] 开始生成过渡音乐...")
        generate_start = time.time()
        success, processed_audio_path, raw_audio_path = service.generate_transition_music(
            from_emotion=from_emotion,
            to_emotion=to_emotion,
            from_prompt=from_prompt,
            to_prompt=to_prompt,
            transition_duration=transition_duration
        )
        generate_end = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        if success:
            app.logger.info(f"[{timestamp}] 过渡音乐生成成功，耗时: {generate_end - generate_start:.2f}秒")
            app.logger.info(f"[{timestamp}] 音频文件路径: processed={processed_audio_path}, raw={raw_audio_path}")
            
            # 转换为相对路径
            if processed_audio_path:
                processed_audio_file = os.path.basename(processed_audio_path)
                processed_audio_url = url_for('static', filename=f'generated_music/{processed_audio_file}')
            else:
                processed_audio_url = None
                
            if raw_audio_path:
                raw_audio_file = os.path.basename(raw_audio_path)
                raw_audio_url = url_for('static', filename=f'generated_music/{raw_audio_file}')
            else:
                raw_audio_url = None
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.info(f"[{timestamp}] 音频URL: processed={processed_audio_url}, raw={raw_audio_url}")
            
            response = {
                "status": "success",
                "message": "过渡音乐生成成功",
                "processed_audio": processed_audio_url,
                "raw_audio": raw_audio_url,
                "from_emotion": from_emotion,
                "to_emotion": to_emotion,
                "transition_duration": transition_duration
            }
            
            overall_end_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.info(f"[{timestamp}] 请求处理成功，总耗时: {overall_end_time - overall_start_time:.2f}秒")
            app.logger.info(f"[{timestamp}] ===== 情绪过渡音乐生成请求处理完成 =====")
            return jsonify(response)
        else:
            app.logger.error(f"[{timestamp}] 过渡音乐生成失败，耗时: {generate_end - generate_start:.2f}秒")
            
            response = {
                "status": "error",
                "message": "过渡音乐生成失败，请稍后再试",
                "error": "音乐生成服务出错"
            }
            
            overall_end_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            app.logger.error(f"[{timestamp}] 请求处理失败，总耗时: {overall_end_time - overall_start_time:.2f}秒")
            app.logger.error(f"[{timestamp}] ===== 情绪过渡音乐生成请求处理失败 =====")
            return jsonify(response), 500
            
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        app.logger.error(f"[{timestamp}] 处理过渡音乐请求时发生错误: {str(e)}")
        app.logger.error(f"[{timestamp}] 错误堆栈: {traceback.format_exc()}")
        
        overall_end_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        app.logger.error(f"[{timestamp}] 请求处理异常，总耗时: {overall_end_time - overall_start_time:.2f}秒")
        app.logger.error(f"[{timestamp}] ===== 情绪过渡音乐生成请求处理异常 =====")
        
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {str(e)}",
            "error": str(e)
        }), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # 获取原始音频路径和处理参数
        raw_audio_path = request.json.get('raw_audio_path')
        emotion_label = request.json.get('emotion_label')
        eq_params = request.json.get('eq_params')
        target_db = request.json.get('target_db', -20.0)
        limiter_threshold = request.json.get('limiter_threshold', -1.0)
        
        if not raw_audio_path:
            return jsonify({'success': False, 'error': '未找到原始音频'})
        
        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音频处理服务不可用，请联系管理员'})
            
        # 如果提供了情绪标签但没有均衡器参数，则使用情绪预设的参数    
        if emotion_label and emotion_label in EMOTION_PRESET_PARAMS and not eq_params:
            audio_params = service.get_audio_params_for_emotion(emotion_label)
            eq_params = {
                'low_shelf_gain': audio_params['low_shelf_gain'],
                'mid_gain': audio_params['mid_gain'],
                'high_shelf_gain': audio_params['high_shelf_gain']
            }
            target_db = audio_params['target_db']
            limiter_threshold = audio_params['limiter_threshold']
            
        # 构建完整的文件路径
        full_raw_path = os.path.join(app.static_folder, raw_audio_path)
        
        # 读取原始音频
        audio_data, sample_rate = sf.read(full_raw_path)
        
        # 处理音频
        processed_audio = service.audio_processor.process_audio(
            audio_data,
            normalize=True,
            target_db=target_db,
            eq_params=eq_params
        )
        processed_audio = service.audio_processor.apply_limiter(processed_audio, limiter_threshold)
        
        # 生成新的处理后音频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"processed_music_{timestamp}.wav"
        processed_path = os.path.join(app.static_folder, 'generated_music', processed_filename)
        
        # 保存处理后的音频
        sf.write(processed_path, processed_audio, sample_rate)
        
        # 生成音频文件路径
        processed_audio_file = f'generated_music/{processed_filename}'
        raw_audio_file = f'generated_music/{raw_audio_file}'
        
        # 使用 url_for 生成正确的 URL
        processed_audio_url = url_for('static', filename=processed_audio_file)
        raw_audio_url = url_for('static', filename=raw_audio_file)
        
        return jsonify({
            'success': True,
            'audio_path': processed_audio_url,
            'raw_audio_path': raw_audio_url,
            'emotion_label': emotion_label,
            'stage': 'processed'  # 添加阶段标识
        })
        
    except Exception as e:
        print(f"音频处理错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/transition')
def transition():
    return render_template('transition.html')

if __name__ == '__main__':
    # 确保静态文件目录存在
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    generated_music_dir = os.path.join(static_dir, 'generated_music')
    Path(generated_music_dir).mkdir(parents=True, exist_ok=True)
    app.run(debug=True) 