#!/bin/bash

# ============================================================
# 虚假图像检测训练/测试启动脚本
# ============================================================

set -e  # 遇错停止

# ==================== 配置文件路径 ====================
yaml_config="${yaml_config:-D:/模拟桌面/科研/cv/ForensicArtifacts/config/noncurriculum.yaml}"

# 检查配置文件存在
if [ ! -f "$yaml_config" ]; then
    echo "❌ 配置文件不存在: $yaml_config"
    exit 1
fi

echo "=========================================="
echo "  使用配置文件: $yaml_config"
echo "=========================================="

# ==================== 解析YAML配置 ====================
# 读取关键参数
gpus=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['gpus'])")
flag=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['flag'])")
log_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['log_dir'])")
save_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_config')).get('save_dir', './checkpoints'))")

# 计算GPU数量
gpu_count=$(echo $gpus | awk -F',' '{print NF}')

echo "  使用GPU: $gpus (共 $gpu_count 张)"
echo "  运行模式: $flag"
echo "  日志目录: $log_dir"
echo "  检查点目录: $save_dir"

# ==================== 环境设置 ====================
# 设置Python路径
export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建必要目录
mkdir -p ${log_dir}
mkdir -p ${save_dir}

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${gpus}

# ==================== 选择运行脚本 ====================
if [ "$flag" = "test" ]; then
    script_path="test.py"
    echo " 运行测试脚本: $script_path"
elif [ "$flag" = "train" ]; then
    script_path="train.py"
    echo " 运行训练脚本: $script_path"
else
    echo "❌ 配置文件中的 flag 字段必须是 'test' 或 'train'，当前是 '$flag'"
    exit 1
fi

# ==================== 启动训练/测试 ====================
echo ""
echo "=========================================="
echo "  开始运行..."
echo "=========================================="

# 判断是否使用分布式训练
if [ "$gpu_count" -gt 1 ]; then
    echo "📡 使用分布式训练 (DDP), GPU数量: $gpu_count"
    
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${gpu_count} \
        ${script_path} \
        --config ${yaml_config} \
        2> ${log_dir}/error.log 1> ${log_dir}/train.log
else
    echo "  使用单GPU训练"
    
    python3 ${script_path} \
        --config ${yaml_config} \
        2> ${log_dir}/error.log 1> ${log_dir}/train.log
fi

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "✅ 运行完成!"
echo "  日志文件: ${log_dir}/train.log"
echo "  错误日志: ${log_dir}/error.log"
echo "=========================================="