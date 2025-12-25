import os
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

print("=" * 60)
print("RWKV 模型测试")
print("=" * 60)

# 模型路径
model_path = '/workspace/model/rwkv7-g1a4-2.9b-20251118-ctx8192'

# 检查文件
if not os.path.exists(model_path+".pth"):
    print(f"✗ 模型文件不存在: {model_path}")
#    exit(1)

print(f"✓ 模型文件: {model_path}")
print(f"✓ 文件大小: {os.path.getsize(model_path+'.pth') / (1024**3):.2f} GB")

# 加载模型
print("\n加载模型...")
model = RWKV(model=model_path, strategy='cuda fp16')
print("✓ 模型加载成功")

# 创建 pipeline
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# 测试生成
print("\n测试生成...")
prompt = "Hello, I am"
#output = model.generate(pipeline.encode(prompt), 20, temperature=1.0, top_p=0.7)
#generated = pipeline.decode(output)
prompt = "你好，请介绍一下自己 不是生成小说"
output = pipeline.generate(prompt, token_count=1000)
print(f"\nPrompt: {prompt}")
print(f"Generated: {output}")

print("\n" + "=" * 60)
print("✓ 测试完成")
print("=" * 60)
