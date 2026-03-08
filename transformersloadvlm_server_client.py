"""
client 发起请求

# 如果未来有视频处理请求，继续此对话 https://claude.ai/chat/5ec7d36b-09ba-48ae-a314-dc8fcbe8276e
"""

import base64
import requests

# 读取图像并转为 base64
with open("/localmodels/picturesdemo/test.jpeg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 构建请求
response = requests.post(
    "http://localhost:8002/v1/chat/completions",
    json={
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }}
            ]
        }]
    }
)

print(response.json())
