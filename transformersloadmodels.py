from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import json

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
    messages = data['messages']
    
    # 构建输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成回复
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    
    # 只提取新生成的部分（去掉输入prompt）
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 返回结果，确保中文正确显示
    result = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response
            }
        }]
    }
    
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)