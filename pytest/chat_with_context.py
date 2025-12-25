# import os, copy, types, gc, sys, re  # 导入操作系统、对象复制、类型、垃圾回收、系统、正则表达式等包
# import torch # 导入 pytorch 库
# import numpy as np  # 导入 numpy 库
# os.environ['RWKV_JIT_ON'] = '0'
# os.environ['RWKV_CUDA_ON'] = '0'

# from rwkv.model import RWKV
# from rwkv.utils import PIPELINE

# print(" 加载模型中...")
# model = RWKV(model='/workspace/model/rwkv7-g1a4-2.9b-20251118-ctx8192', strategy='cuda fp16')
# pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# print(" 模型加载完成！")
# print(" 支持多轮对话，会记住上下文")
# print(" 输入 'clear' 清空对话历史")
# print(" 输入 'exit' 退出\n")

# # 对话历史
# conversation_history = []

# while True:

#     user_input = input("你: ")
    
#     if user_input.lower() in ['exit', 'quit', '退出']:
#         print(" 再见！")
#         break
    
#     if user_input.lower() in ['clear', '清空']:
#         conversation_history = []
#         print("  对话历史已清空\n")
#         continue
    
#     if not user_input.strip():
#         continue
    
#     # 添加用户输入到历史
#     conversation_history.append(f"User: {user_input}")
#     # 构建完整上下文
#     # context = "\n".join(conversation_history) + "\nAssistant:<think"
#     context = "\n".join(f"User: {user_input}") + "\nAssistant:<think"
    
#     # 生成回复
#     print(" 思考中...", end='\r')
#     response = pipeline.generate(context, token_count=400)
    
#     # 清理输出（移除可能的前缀）
#     response = response.strip()
#     if response.startswith("Assistant:<think"):
#         response = response[10:].strip()
    
#     # 添加回复到历史
#     conversation_history.append(f"Assistant:<think {response}")
    
#     print(f"AI: {response}\n")
    
#     # 限制历史长度（防止超出上下文窗口）
#     if len(conversation_history) > 20:
#         conversation_history = conversation_history[-20:]

import os, copy, types, gc, sys, re  # 导入操作系统、对象复制、类型、垃圾回收、系统、正则表达式等包
import torch # 导入 pytorch 库
import numpy as np  # 导入 numpy 库
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_CUDA_ON'] = '0'


from rwkv.model import RWKV
from rwkv.utils import PIPELINE

class ChatBot:
    def __init__(self, model_path):
        print("🤖 加载模型中...")
        self.model = RWKV(model=model_path, strategy='cuda fp16')
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.conversation = []
        print("✅ 模型加载完成！\n")
    
    def generate_response(self, prompt, max_tokens=400):
        """手动生成，只返回新内容"""
        # 编码输入
        tokens = self.pipeline.encode(prompt)
        state = None
        
        # 处理输入 tokens（不输出）
        for token in tokens:
            _, state = self.model.forward([token], state)
        
        # 生成新内容
        output_tokens = []
        generated_text = ""
        
        print("AI: ", end='', flush=True)
        
        for i in range(max_tokens):
            # 前向传播
            if i == 0 and len(tokens) > 0:
                out, state = self.model.forward([tokens[-1]], state)
            else:
                out, state = self.model.forward([output_tokens[-1]], state)
            
            # 采样（只使用兼容的参数）
            token = self.pipeline.sample_logits(
                out, 
                temperature=1.0,  # 控制随机性：0.8 更保守，1.2 更随机
                top_p=0.7,        # 核采样：0.7-0.9 都不错
                top_k=0           # 0 表示不使用 top_k
            )
            
            # 检查结束符
            if token == 0:
                break
            
            output_tokens.append(token)
            
            # 实时解码并输出
            tmp = self.pipeline.decode(output_tokens)
            if '\ufffd' not in tmp:
                new_part = tmp[len(generated_text):]
                print(new_part, end='', flush=True)
                generated_text = tmp
            
            # 检测停止标记
            stop_marks = ['\nUser:', '\n你:', '\nHuman:', 'User:', '你:']
            if any(mark in generated_text for mark in stop_marks):
                for mark in stop_marks:
                    if mark in generated_text:
                        generated_text = generated_text.split(mark)[0]
                        break
                break
        
        print("\n")
        return generated_text.strip()
    
    def chat(self, user_input):
        """处理用户输入并生成回复"""
        # 1. 添加用户输入到历史
        self.conversation.append(f"User: {user_input}")
        
        # 2. 构建上下文（使用最近 10 轮对话）
        recent = self.conversation[-20:]
        context = "\n".join(recent) + "\nAssistant: <think>"
        
        # 3. 生成回复（只获取新生成的部分）
        response = self.generate_response(context, max_tokens=800)
        
        # 4. 清理回复
        response = response.replace("</think>", "").strip()
        if response.startswith("<think>"):
            response = response[7:].strip()
        
        # 5. 保存到历史
        self.conversation.append(f"Assistant: <think> {response}")
        
        # 6. 限制历史长度
        if len(self.conversation) > 40:
            self.conversation = self.conversation[-40:]
        
        return response
    
    def clear(self):
        self.conversation = []
        print("🗑️  对话历史已清空\n")
    
    def show_history(self):
        print("\n" + "="*60)
        print("📝 对话历史 (共 {} 条):".format(len(self.conversation)))
        print("="*60)
        for i, line in enumerate(self.conversation, 1):
            print(f"{i}. {line}")
        print("="*60 + "\n")


def main():
    model_path = '/workspace/model/rwkv7-g1a4-2.9b-20251118-ctx8192'
    chatbot = ChatBot(model_path)
    
    print("💬 RWKV 交互式对话")
    print("="*60)
    print("命令:")
    print("  exit/quit    - 退出")
    print("  clear        - 清空历史")
    print("  history      - 查看历史")
    print("="*60)
    print()
    
    while True:
        try:
            user_input = input("你: ")
            
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 再见！")
                break
            
            if user_input.lower() in ['clear', '清空']:
                chatbot.clear()
                continue
            
            if user_input.lower() in ['history', '历史']:
                chatbot.show_history()
                continue
            
            if not user_input.strip():
                continue
            
            # 生成回复
            chatbot.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()