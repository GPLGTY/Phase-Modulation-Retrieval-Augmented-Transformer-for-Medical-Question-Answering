
# coding=utf-8
# Copyright 2024 South China University of Technology and 
# Engineering Research Ceter of Ministry of Education on Human Body Perception.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn or 1132461715@qq.com>
# Date: 2024.03.24

# File Name: openai_api_llm.py
# Description: 模型调用统一框架

# 使用说明
# 需要在~/.bashrc当中添加相应的环境变量并source
'''

# 设置访问https://portal.azure.com提供的ChatGPT接口服务的api_key变量
# https://zhishenggpt.openai.azure.com/
export GPT35_AZURE_OPENAI_KEY='xxxxxxx'
# https://zhishenggpt40.openai.azure.com/
export GPT4_AZURE_OPENAI_KEY='xxxxxx'

# https://openai.api2d.net/v1
export API2D_OPENAI_KEY='fk204884-xxxxx'

# https://dev.iai007.cloud/ai/api/v1
export HEFEI_OPENAI_KEY='xxxxxx'

'''

import os
import time
from typing import Literal
from openai import OpenAI, AzureOpenAI


class OpenAI_LLM:
    '''
    功能：支持下列模型
        微软Azure(https://portal.azure.com):
            'gpt-3.5-turbo-1106_azure': 'GPT-35', 
            'gpt-4-1106-preview_azure': 'GPT4', 
        API2D(https://api2d.com/wiki/doc):
            'gpt-3.5-turbo-1106_api2d': 'gpt-3.5-turbo-1106',
            'gpt-3.5-turbo-16k-0613_api2d': 'gpt-3.5-turbo-16k-0613', 
            'gpt-3.5-turbo-16k_api2d': 'gpt-3.5-turbo-16k', 
            'gpt-3.5-turbo-0613_api2d': 'gpt-3.5-turbo-0613', 
            'gpt-3.5-turbo_api2d': 'gpt-3.5-turbo', 
            'gpt-3.5-turbo-0301_api2d': 'gpt-3.5-turbo-0301', 
            'gpt-4-1106-preview_api2d': 'gpt-4-1106-preview',
            'gpt-4-0613_api2d': 'gpt-4-0613', 
            'gpt-4_api2d': 'gpt-4',
        合肥开发团队:
            'qwen-72b-chat_hefei': 'qwen-72b-chat',
            'qwen-14b-chat_hefei': 'qwen-14b-chat',
    '''

    def __init__(self, model_name, system_prompt=None):
        '''
        输入格式：
            model_name: 字符串，表示模型的名称（见功能说明）。
                        当前支持：gpt-3.5-turbo-1106_azure, gpt-4-1106-preview_azure, gpt-3.5-turbo-1106_api2d, gpt-4-1106-preview_api2d, gpt-3.5-turbo-16k-0613_api2d, gpt-3.5-turbo-16k_api2d, 
                        gpt-3.5-turbo-0613_api2d, gpt-3.5-turbo_api2d, gpt-4-0613_api2d, gpt-4_api2d, qwen-72b-chat_hefei, qwen-14b-chat_hefei
                        moonshot-v1-32k_kimi, moonshot-v1-128k_kimi, moonshot-v1-8k_kimi,
            system_prompt: 字符串，表示系统的指令说明，定义了'role': 'system'对应的设定内容
                           当其为None时，调用默认的设置初始化
        '''
        self.model_name = model_name
        self.system_prompt = system_prompt
        if model_name.endswith("_hefei"):
            self.client = OpenAI(
                base_url="https://dev.iai007.cloud/ai/api/v1",
                api_key=os.getenv("HEFEI_OPENAI_KEY"),
            )
            self.model = model_name.lower().split("_")[0]

        elif model_name.endswith("_api2d"):
            self.client = OpenAI(
                base_url="https://openai.api2d.net/v1",
                api_key=os.getenv("API2D_OPENAI_KEY"),
            )
            self.model = model_name.split("_")[0]
            if system_prompt is None:
                if "gpt-4" in model_name:
                    now = time.localtime()
                    current_date = time.strftime("%Y-%m", now)
                    self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-04\nCurrent date: {current_date}'
                elif "gpt-3.5" in model_name:
                    now = time.localtime()
                    current_date = time.strftime("%Y-%m", now)
                    
                    self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent date: {current_date}'


        elif model_name.endswith("_azure"):
            '''
            https://portal.azure.com/
            # 账号密码请联系陈艺荣

            model_deployment_name_on_azure="GPT35" or "GPT4" 
            # 需要确定具体的部署版本号请联系陈艺荣
            # GPT4: gpt-4-1106-Preview, 输入：128,000 输出：4,096, 训练数据（上限）：2023年4月
            # GPT-35: gpt-35-turbo-1106, 输入：16,385 输出：4,096, 训练数据（上限）：2021年9月
            '''
            if "gpt-4" in model_name:
                self.client = AzureOpenAI(
                    azure_endpoint="https://zhishenggpt40.openai.azure.com/",
                    api_key=os.getenv("GPT4_AZURE_OPENAI_KEY"),
                    api_version="2024-02-15-preview",
                )
                self.model = "GPT4"
                if system_prompt is None:
                    now = time.localtime()
                    current_date = time.strftime("%Y-%m", now)
                    self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-04\nCurrent date: {current_date}'
            elif "gpt-3.5" in model_name:
                self.client = AzureOpenAI(
                    azure_endpoint="https://zhishenggpt.openai.azure.com/",
                    api_key=os.getenv("GPT35_AZURE_OPENAI_KEY"),
                    api_version="2024-02-15-preview",
                )
                self.model = "GPT-35"
                if system_prompt is None:
                    now = time.localtime()
                    current_date = time.strftime("%Y-%m", now)
                    self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent date: {current_date}'

            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            
        elif model_name.endswith("_kimi"):
            # moonshot-v1-8k_kimi、moonshot-v1-32k_kimi、moonshot-v1-128k_kimi
            self.system_prompt = '你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。'
            self.client = OpenAI(
                base_url="https://api.moonshot.cn/v1",
                api_key=os.getenv("KIMI_OPENAI_KEY"),  # 
            )
            self.model = model_name.split("_")[0]

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def chat(
        self,
        messages: list[dict[str, str]],
        generation_config = None,
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        add_system_prompt = True,
        return_completions = True
    ):
        '''
        功能：通用的chat接口
        输入格式：
            messages: 一个字典列表，看起来像这样：
                [
                    {
                        'role': 'system', # 'system' or 'assistant' or 'user'
                        'content': <str>
                    
                    },
                    ...
                
                ]
            
            generation_config: 传入的生成配置，该参数位了兼容本地部署的大模型调用
            temperature: 温度系数
            max_tokens: 最大的输出token数
            stream: 控制是否流式返回
            add_system_prompt: 强制添加system_prompt
            return_completions: 强制返回completions形式，当该参数为True时，无论stream是否为True都以completions形式返回
            
                
                
        
        '''
        if add_system_prompt and self.system_prompt is not None:
            # 强制检查系统Prompt并且添加到messages的开头
            if self.model_name.endswith("_api2d"):
                if messages[0]["role"] != "system":
                    # 如果传入的messages不存在system_prompt，则添加system_prompt
                    messages = [{"role":"system","content":self.system_prompt}] + messages # 拼接system_prompt
            
            elif self.model_name.endswith("_azure"):
                if messages[0]["role"] != "system":
                    # 如果传入的messages不存在system_prompt，则添加system_prompt
                    messages = [{"role":"system","content":self.system_prompt}] + messages # 拼接system_prompt

            else:
                if messages[0]["role"] != "system":
                    # 如果传入的messages不存在system_prompt，则添加system_prompt
                    messages = [{"role":"system","content":self.system_prompt}] + messages # 拼接system_prompt
        
        completions = self.client.chat.completions.create(
            model=self.model, 
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream # 流式返回
        )

        if stream == False and return_completions == False:
            # 直接返回字符串
            return completions.choices[0].message.content
        else:
            # 流式返回或者要求返回completions
            return completions