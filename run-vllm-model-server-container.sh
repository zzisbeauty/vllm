#!/bin/bash  

# 此 sh 脚本是使用 vllm 官方镜像，直接启动服务; 用方法: ./switch-vllm-model.sh <模型名称>  
#                               例如: ./switch-vllm-model.sh qwen2.5-0.5b-instruct  
#                               这种方式更换模型，需要删除原来的容器，基于新模型重启新容器

set -e

# 配置变量
CONTAINER_NAME="vllm-server-base-official-image"    # 启动的容器名称
LOCAL_MODELS_DIR="/home/nvidia/Desktop/volumes-models"  
CONTAINER_MODELS_DIR="/app/models"  
PORT=1127

# 检查是否提供了模型名称  
if [ -z "$1" ]; then  
    echo "错误: 请提供模型名称"  
    echo "使用方法: $0 <模型名称>"  
    echo "例如: $0 qwen2.5-0.5b-instruct"  
    exit 1  
fi

MODEL_NAME=$1  
MODEL_PATH="${LOCAL_MODELS_DIR}/${MODEL_NAME}"  

# 检查模型目录是否存在  
if [ ! -d "$MODEL_PATH" ]; then  
    echo "错误: 模型目录不存在: $MODEL_PATH"  
    echo "可用的模型:"  
    ls -1 "$LOCAL_MODELS_DIR"  
    exit 1  
fi  

echo "正在切换到模型: $MODEL_NAME"  

# 停止并删除旧容器
echo "停止旧容器..."  
docker stop "$CONTAINER_NAME" 2>/dev/null || true  
docker rm "$CONTAINER_NAME" 2>/dev/null || true  
  
# 启动新容器  
echo "启动新容器..."  
docker run -d \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "${LOCAL_MODELS_DIR}:${CONTAINER_MODELS_DIR}" \
    -p ${PORT}:8000 \
    --ipc=host \
    --name "$CONTAINER_NAME" \
    vllm/vllm-openai:latest \
    --model "${CONTAINER_MODELS_DIR}/${MODEL_NAME}" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --max-num-seqs 128

# --max-num-batched-tokens 10000
# 增加 max_num_seqs: 提高吞吐量(可以处理更多并发请求),但会增加每个请求的端到端延迟
# 减少 max_num_seqs: 降低延迟,但会限制并发处理能力
# max_num_seqs 与 max_num_batched_tokens 配合使用来控制批处理大小，在自动调优脚本中,这两个参数通常一起调整以找到最佳的吞吐量配置

echo "等待30s容器启动..."
sleep 30

# 检查容器状态  
if docker ps | grep -q "$CONTAINER_NAME"; then  
    echo "✓ 容器启动成功!"  
    echo "模型: $MODEL_NAME"  
    echo "端口: http://localhost:${PORT}"  
    echo ""
    echo "查看日志: docker logs -f $CONTAINER_NAME"  
else  
    echo "✗ 容器启动失败"  
    echo "查看错误日志: docker logs $CONTAINER_NAME"  
    exit 1  
fi
