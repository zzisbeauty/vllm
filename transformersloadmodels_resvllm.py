"""
响应结果和 vllm 一致的情况

同步处理请求，阻塞
"""


from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import json
import time
import uuid

app = Flask(__name__)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/localmodels/Qwen3-4B-Instruct-2507",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("/localmodels/Qwen3-4B-Instruct-2507")

@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    model_name = data.get('model', 'Qwen3-4B-Instruct-2507')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    stream = data.get('stream', False)
    
    # 构建输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    prompt_tokens = inputs['input_ids'].shape[1]
    
    # 生成回复
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 只提取新生成的部分
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    completion_tokens = len(generated_tokens)
    total_tokens = prompt_tokens + completion_tokens
    
    # 构建 vLLM 兼容的响应
    result = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }
    
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False, indent=2),
        status=200,
        mimetype='application/json'
    )

@app.route('/v1/models', methods=['GET'])
def models():
    """兼容 vLLM 的 /v1/models 接口"""
    result = {
        "object": "list",
        "data": [
            {
                "id": "Qwen3-4B-Instruct-2507",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
