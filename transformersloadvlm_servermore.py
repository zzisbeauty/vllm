"""
多模态模型的更新完善的服务端代码

但是这个版本没测试，没使用，目前用的这个服务 transformersloadvlm_server.py

如果未来用了有问题，继续这个对话  https://claude.ai/chat/5ec7d36b-09ba-48ae-a314-dc8fcbe8276e
"""

from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import json
import time
import uuid
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# 加载模型
model = AutoModel.from_pretrained(
    "/localmodels/MiniCPM-V-2_6-int4",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    "/localmodels/MiniCPM-V-2_6-int4",
    trust_remote_code=True
)

def decode_image(image_data):
    """解码图像：支持文件路径和 base64"""
    # 情况1: 文件路径
    if isinstance(image_data, str) and os.path.exists(image_data):
        return Image.open(image_data)
    
    # 情况2: base64 编码
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    try:
        image_bytes = base64.b64decode(image_data)
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"无法解码图像: {e}")

@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    
    # 提取图像
    image = None
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url:
                        try:
                            image = decode_image(image_url)
                            break
                        except Exception as e:
                            return jsonify({"error": f"图像加载失败: {str(e)}"}), 400
    
    # 构建文本消息
    text_messages = []
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, str):
            text_messages.append({"role": msg['role'], "content": content})
        elif isinstance(content, list):
            text_content = ' '.join([
                item.get('text', '') 
                for item in content 
                if item.get('type') == 'text'
            ])
            if text_content:
                text_messages.append({"role": msg['role'], "content": text_content})
    
    try:
        # 调用模型
        response_text = model.chat(
            image=image,
            msgs=text_messages,
            tokenizer=tokenizer
        )
        
        # 构建响应
        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "MiniCPM-V-2_6-int4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False, indent=2),
            status=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1128, threaded=True)