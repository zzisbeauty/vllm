from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import json
import time
import uuid
from queue import Queue
from threading import Thread, Lock

app = Flask(__name__)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/localmodels/Qwen3-4B-Instruct-2507",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("/localmodels/Qwen3-4B-Instruct-2507")

# 请求队列和结果存储
request_queue = Queue()
result_store = {}
store_lock = Lock()

def inference_worker():
    """后台推理线程，处理队列中的请求"""
    while True:
        request_id, messages, params = request_queue.get()
        
        try:
            # 构建输入
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            prompt_tokens = inputs['input_ids'].shape[1]
            
            # 生成回复
            outputs = model.generate(
                **inputs,
                max_new_tokens=params['max_tokens'],
                do_sample=params['temperature'] > 0,
                temperature=params['temperature'] if params['temperature'] > 0 else 1.0,
                top_p=params['top_p'],
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 提取生成内容
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            completion_tokens = len(generated_tokens)
            
            # 构建结果
            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": params['model_name'],
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            # 存储结果
            with store_lock:
                result_store[request_id] = {"status": "completed", "result": result}
                
        except Exception as e:
            with store_lock:
                result_store[request_id] = {"status": "error", "error": str(e)}
        
        request_queue.task_done()

# 启动后台推理线程
inference_thread = Thread(target=inference_worker, daemon=True)
inference_thread.start()

@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    
    # 生成请求ID
    request_id = uuid.uuid4().hex
    
    # 提取参数
    params = {
        'model_name': data.get('model', 'Qwen3-4B-Instruct-2507'),
        'max_tokens': data.get('max_tokens', 512),
        'temperature': data.get('temperature', 0.7),
        'top_p': data.get('top_p', 0.9)
    }
    
    # 初始化结果存储
    with store_lock:
        result_store[request_id] = {"status": "pending"}
    
    # 加入队列
    request_queue.put((request_id, messages, params))
    
    # 轮询等待结果（带超时）
    timeout = 300  # 5分钟超时
    start_time = time.time()
    
    while True:
        with store_lock:
            if request_id in result_store:
                status_data = result_store[request_id]
                
                if status_data["status"] == "completed":
                    result = status_data["result"]
                    del result_store[request_id]  # 清理
                    return app.response_class(
                        response=json.dumps(result, ensure_ascii=False, indent=2),
                        status=200,
                        mimetype='application/json'
                    )
                
                elif status_data["status"] == "error":
                    error = status_data["error"]
                    del result_store[request_id]
                    return jsonify({"error": error}), 500
        
        # 超时检查
        if time.time() - start_time > timeout:
            with store_lock:
                if request_id in result_store:
                    del result_store[request_id]
            return jsonify({"error": "Request timeout"}), 504
        
        time.sleep(0.1)  # 避免CPU空转

@app.route('/v1/models', methods=['GET'])
def models():
    result = {
        "object": "list",
        "data": [{
            "id": "Qwen3-4B-Instruct-2507",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    }
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "queue_size": request_queue.qsize()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)


