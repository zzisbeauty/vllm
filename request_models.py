from openai import OpenAI  

# Completions API  
client = OpenAI(  
    api_key="EMPTY",  
    base_url="http://localhost:8000/v1",  
)  
  
completion = client.completions.create(  
    model="qwen2.5-0.5b-instruct",  
    prompt="San Francisco is a",  
)  
print("Completion result:", completion)


# Chat Completions API
client = OpenAI(  
    api_key="EMPTY",  
    base_url="http://localhost:8000/v1",  
)  
  
chat_response = client.chat.completions.create(  
    model="qwen2.5-0.5b-instruct",  
    messages=[  
        {"role": "system", "content": "You are a helpful assistant."},  
        {"role": "user", "content": "Tell me a joke."},  
    ],  
    stream=True, # 对话的流式响应
)  
print("Chat response:", chat_response)