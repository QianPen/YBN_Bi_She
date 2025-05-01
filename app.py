from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import torch
from werkzeug.utils import secure_filename
from evaluate_audio import load_model, evaluate_single_audio
from config import Config
from flask_assets import Environment, Bundle

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置静态文件
assets = Environment(app)
assets.url = app.static_url_path

# 配置 CSS 和 JS 文件
css = Bundle(
    'css/style.css',
    filters='cssmin',
    output='gen/packed.css'
)
js = Bundle(
    'js/main.js',
    filters='jsmin',
    output='gen/packed.js'
)
assets.register('css_all', css)
assets.register('js_all', js)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

# 加载模型
model = load_model('saved_models/best_model.pth')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate', methods=['POST'])
def evaluate_audio():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    # 检查文件是否为空
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 评估音频
        result = evaluate_single_audio(filepath, model)
        
        # 删除临时文件
        os.remove(filepath)
        
        # 返回结果
        if Config.TASK_TYPE == 'regression':
            return jsonify({
                'score': float(result),
                'message': '评估完成'
            })
        else:
            return jsonify({
                'class': int(result),
                'message': '评估完成'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 