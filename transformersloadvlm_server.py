from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import json
import time
import uuid
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# 加载 VLM 模型
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

# def decode_image(image_data):
#     """解码 base64 图像"""
#     if image_data.startswith('data:image'):
#         image_data = image_data.split(',')[1]
#     image_bytes = base64.b64decode(image_data)
#     return Image.open(BytesIO(image_bytes))

def decode_image(image_data):
    """解码图像：支持 base64 和文件路径"""
    # 如果是文件路径
    if image_data.startswith('/') or image_data.startswith('./'):
        return Image.open(image_data)
    
    # 如果是 base64
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes))


@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    
    # 提取图像（如果有）
    image = None
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url:
                        image = decode_image(image_url)
                        break
    
    # 构建纯文本消息
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
        if image:
            response_text = model.chat(
                image=image,
                msgs=text_messages,
                tokenizer=tokenizer
            )
        else:
            # 纯文本对话（如果模型支持）
            response_text = model.chat(
                image=None,
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "MiniCPM-V-2_6-int4"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, threaded=True)


""" 发起请求

# 纯文本
curl -X POST http://localhost:8002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "你好"}
        ]
    }'


# 带图像（OpenAI Vision API 格式）

# 情况 - 1 本地传入 picture
curl -X POST http://localhost:8002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {"type": "image_url", "image_url": {"url": "/localmodels/picturesdemo/test.jpeg"}}
            ]
        }]
    }'

# 情况 - 2 远程传入 base64  -   transformersloadvlm_server_client.py
"""