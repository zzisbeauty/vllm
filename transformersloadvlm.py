"""
官方模式加载代码， 多模态模型测试
"""

from transformers import AutoModel, AutoTokenizer
import torch

# 加载模型（视觉语言模型）
model = AutoModel.from_pretrained(
    "/localmodels/MiniCPM-V-2_6-int4",
    trust_remote_code=True,  # 重要！VLM 通常需要自定义代码
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    "/localmodels/MiniCPM-V-2_6-int4",
    trust_remote_code=True
)

# 测试（需要图像输入）
from PIL import Image

image = Image.open("/localmodels/picturesdemo/test.jpeg")
messages = [
    {"role": "user", "content": "描述这张图片"}
]

# VLM 的推理方式不同
response = model.chat(
    image=image,
    msgs=messages,
    tokenizer=tokenizer
)
print(response)