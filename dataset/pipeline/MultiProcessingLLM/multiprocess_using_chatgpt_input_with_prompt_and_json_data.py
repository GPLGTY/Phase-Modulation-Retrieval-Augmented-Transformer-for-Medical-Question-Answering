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

# File Name: multiprocess_using_chatgpt_input_with_prompt_and_json_data.py
# Description: generate chatgpt answer
#              利用ChatGPT进行批量处理

# Reference: 
# [1] https://learn.microsoft.com/zh-cn/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python


# 使用示例
'''
## 安装依赖
pip install openai==1.7.1

## 运行示例
conda activate llama_factory

# 生成问题与答案之间的解析
cd dataset/pipeline
python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="gpt-4-1106-preview_api2d" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain_en.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_gpt-4-1106-preview_api2d" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95


'''

import os
import re
import json
import time
import copy
import openai
import argparse
import tiktoken
import datetime
import multiprocessing
from openai_api_llm import OpenAI_LLM

# 公共配置
chatgpt_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
gpt_4_encoding = tiktoken.encoding_for_model("gpt-4")
llm_azure_gpt4 = OpenAI_LLM(model_name='gpt-4-1106-preview_azure') # 当调用微软GPT-3.5时遇到超长文本时调用
llm_api2d_gpt35 = OpenAI_LLM(model_name='gpt-3.5-turbo-1106_api2d') # 当调用微软GPT-3.5时遇到安全过滤时调用
llm_api2d_gpt4 = OpenAI_LLM(model_name='gpt-4-1106-preview_api2d') # 当调用微软GPT-3.5时遇到安全过滤时且输入文本过长时调用


# 公共模板
def get_element_from_dict_with_index_list(temp_dict, index_list):
    '''
    功能：获取复杂字典中指定index的元素
    输入格式:
        temp_dict: 字典，参考格式：
            {
                'id': 'xxx',
                'name':  <str>,
                'question': <str>,
                'answer_list':
                    [
                        {
                            'content': <str>,
                        },
                        ...
                    ]
            }
        index_list: 一个一维列表
            ['answerOptions', 1, 'answerUserContent'] # 读取temp_dict['answerOptions'][1]['answerUserContent']
    '''
    try:
        element = None
        for i in range(len(index_list)):
            if i == 0:
                element = temp_dict[index_list[i]]
            else:
                element = element[index_list[i]]
        
        return element
                
    except KeyError as e:
        print(f"【错误返回】输入的index_list存在键或者下标不在字典temp_dict的范围中: {e}")


def prompt_from_dict_with_key_names(temp_dict, 
                                    general_prompt='''这是一个包含指定位置插入format字符串的prompt{}''', 
                                    key_names=[]):
    '''
    功能：按照key_names列表提供的键依次读取temp_dict对应的字符串，构建模板，返回一个长字符串
    输入格式:
        temp_dict: 字典，参考格式：
            {
                'id': 'xxx',
                'name':  <str>,
                'question': <str>,
                'answer_list':
                    [
                        {
                            'content': <str>,
                        },
                        ...
                    ]
            }
        general_prompt: 一个包含format插值符{}的字符串
        key_names: 一个二维列表列表
            [
                ['questionTitle'], # 读取temp_dict['questionTitle']
                ['questionDetail'], # 读取temp_dict['questionDetail']
                ['answerOptions', 1, 'answerUserContent'] # 读取temp_dict['answerOptions'][1]['answerUserContent']
            ]
    '''

    # 校对key_names元素个数与占位符{}的数量是否一致
    placeholders_count = general_prompt.count('{}')
    key_names_num = len(key_names)
    if key_names_num != placeholders_count:
        raise ValueError('prompt_from_dict_with_key_names函数的key_names的元素个数必需与general_prompt的format占位符数目一样')

    # 根据key_names依次获取对应的temp_dict元素
    element_list = [str(get_element_from_dict_with_index_list(temp_dict, index_list)) for index_list in key_names]

    # format函数格式化字符串
    total_prompt = general_prompt.format(*element_list) # 在Python中，你可以使用*操作符将列表的所有元素传入format函数。这个操作符会将列表中的元素作为参数传递给函数。

    return total_prompt


def load_raw_json_data(data_path):
    '''
    功能：读取'./data/total_raw_data.json'并返回完整的数据raw_json_data

    raw_json_data = {
        'data': [
            {
               ...
            },
            ...
        ]
    }
    
    '''
    try:
        print('【提示】正在从data_path={}指定的文件读取原始数据...'.format(data_path))
        with open(data_path,'r',encoding='utf-8') as f :
            raw_json_data = json.load(f)


        # 如果raw_json_data是列表则直接返回
        if isinstance(raw_json_data, list):
            print('原始数据集总共的样本（求助贴）数=', len(raw_json_data))
            raw_json_data = {'data': raw_json_data}
            return raw_json_data
        
        else:
            # 打印一些基本信息
            print('原始数据集总共的样本（求助贴）数=', len(raw_json_data['data']))
            return raw_json_data

    except FileNotFoundError:
        print('【报错】data_path={}指定的文件不存在，请检查文件路径。'.format(data_path))



def load_raw_json_data_with_process(data_path):
    '''
    每次调用时可以修改该特色函数
    功能：读取'./data/total_raw_data.json'并返回完整的数据raw_json_data

    raw_json_data = {
        'data': [
            {
               ...
            },
            ...
        ]
    }
    
    '''
    try:
        print('【提示】正在从data_path={}指定的文件读取原始数据...'.format(data_path))
        with open(data_path,'r',encoding='utf-8') as f :
            raw_json_data = json.load(f)


        # 如果raw_json_data是列表则直接返回
        if isinstance(raw_json_data, list):
            print('原始数据集总共的样本（求助贴）数=', len(raw_json_data))
            raw_json_data = {'data': raw_json_data}
            #return raw_json_data
        
        else:
            # 打印一些基本信息
            print('原始数据集总共的样本（求助贴）数=', len(raw_json_data['data']))
            #return raw_json_data
        
        # 进行二次处理
        new_raw_json_data = []
        for example in raw_json_data['data']:
            knowledge_lists = example["retrieve_info"]["knowledges_info"]["knowledges"]
            knowledges_for_llm = "\n\n".join(knowledge_lists)
            new_example = example
            new_example["knowledges_for_llm"] = knowledges_for_llm
            new_raw_json_data.append(
                new_example

            )

        return {'data': new_raw_json_data}

    except FileNotFoundError:
        print('【报错】data_path={}指定的文件不存在，请检查文件路径。'.format(data_path))



# 定义一个辅助函数，用于从字符串中提取数字
def extract_number(s):
    numbers = re.findall(r'\d+', s)  # 使用正则表达式找到所有数字
    return int(numbers[0]) if numbers else None


def get_filenames(directory_path, ignore_start_with_string='~', end_with_string="docx", sorted_by_num=False):
    '''获取某个目录下的所有文件名称，忽略~开头的文件
    '''
    # 确保传入的是一个字符串
    if not isinstance(directory_path, str):
        raise ValueError("directory_path must be a string")

    # 获取目录下所有文件的完整路径
    files = os.listdir(directory_path)

    # 筛选出非隐藏文件，即不以~开头的文件
    # 筛选出docx文件，忽略其他格式文件
    filenames = [f for f in files if os.path.isfile(os.path.join(directory_path, f)) and not f.startswith(ignore_start_with_string) and f.endswith(end_with_string)]
    if sorted_by_num:
        sorted_filenames = sorted(filenames, key=extract_number)

        return sorted_filenames
    else:
        return filenames


def txt_to_text(txt_path):
    """读取txt文件并返回字符串"""
    with open(txt_path, "r", encoding="utf-8-sig") as file:
        total_text = file.read()

    return total_text


def load_raw_txt_data(data_dir):
    '''
    功能：读取data_dir中的所有txt文件，并且返回完整的数据raw_json_data

    raw_json_data = {
        'data': [
            {
               ...
            },
            ...
        ]
    }
    
    '''
    # 保存汇总数据
    raw_json_data = {}
    data_list = []

    txt_paths = [os.path.join(data_dir, file_name) for file_name in get_filenames(data_dir, ignore_start_with_string='~', end_with_string="txt", sorted_by_num=False)]

    total_example_num = len(txt_paths)
    for i, txt_path in enumerate(txt_paths):
        txt_file_name = os.path.basename(txt_path)
        file_id = txt_file_name.split('.')[0]
        total_text = txt_to_text(txt_path)

        data_list.append(
                {
                    "id": file_id,
                    "txt_file_name": txt_file_name,
                    "total_text": total_text
                }
            )
        
    raw_json_data['data'] = data_list
    return raw_json_data






def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_list_placeholder_index(key_names_list, list_placeholder='list_placeholder'):
    '''
    功能：返回key_names_list当中list_placeholder指定的占位符元素的下标

    key_names_list: 一个二维列表，元素个数与general_prompt的format占位符数目一致
            其中，字符串为'list_placeholder'的位置代表该处需要进行列表循环
            特别地，最多只允许一处'list_placeholder'
            [
                ['questionTitle'], # 读取temp_dict['questionTitle']
                ['questionDetail'], # 读取temp_dict['questionDetail']
                ['answerOptions', 'list_placeholder', 'answerUserContent'] # 读取temp_dict['answerOptions'][i]['answerUserContent']
            ]
    
    '''
    list_placeholder_i = None
    list_placeholder_j = None
    i = 0
    
    # 循环逐一判断
    for index_names in key_names_list:
        j = 0
        for name in index_names:
            if name == list_placeholder: # 标志位字符串
                list_placeholder_i = i
                list_placeholder_j = j
                return list_placeholder_i, list_placeholder_j
            
            j = j + 1
        
        i = i + 1

    return list_placeholder_i, list_placeholder_j


def add_value_to_dict_with_key_names(temp_dict, key_names= ['answerOptions', 1, 'llm_output_key'], value='测试'):
    '''
    功能：根据key_names在嵌套的字典temp_dict当中添加键值
    
    '''
    current_dict = temp_dict
    for key in key_names[:-1]:
        current_dict = current_dict[key]
    current_dict[key_names[-1]] = value


def get_answer_use_chatgpt(dict_list, 
                           key_names_list,
                           general_prompt,
                           files, 
                           index_key_name='id',
                           output_dir = './output_dir', 
                           model_name='gpt-4-1106-preview_api2d', 
                           list_placeholder = 'list_placeholder',
                           llm_output_key = 'chatgpt_output', # 保存大模型输出的对应的键
                           do_check=False,
                           temperature=0.7, # 大模型生成文本相关参数，下同
                           max_tokens=4096,
                           top_p=0.95,
                           frequency_penalty=0,
                           presence_penalty=0,
                           stop=None,
                           stream=False, # 禁止修改stream设置
                           add_system_prompt=True,
                           return_completions=True): # 禁止修改return_completions设置
    '''
    功能：循环处理dict_list,进行调用ChatGPT生成回复

    输入格式:
        dict_list: 字典列表，参考格式：
            [
                {
                    'id': 'xxx',
                    'name':  <str>,
                    'question': <str>,
                    'answer_list':
                        [
                            {
                                'content': <str>,
                            },
                            ...
                        ]
                }. # 第一个样本
                ...
            
            ]

        key_names_list: 一个二维列表，元素个数与general_prompt的format占位符数目一致
            其中，字符串为'list_placeholder'的位置代表该处需要进行列表循环
            特别地，最多只允许一处'list_placeholder'
            [
                ['questionTitle'], # 读取temp_dict['questionTitle']
                ['questionDetail'], # 读取temp_dict['questionDetail']
                ['answerOptions', 'list_placeholder', 'answerUserContent'] # 读取temp_dict['answerOptions'][i]['answerUserContent']
            ]
        general_prompt: 一个字符串，包含若干个format占位符,例如：
            '问：{}\n答：{}，请润色'
        files: 传入的已经保存在output_dir的文件名称汇总列表（去掉后缀）
        index_key_name: dict_list当中每个样本的标识id的对应键，通常为'id'
        output_dir: 处理完的样本输出的目录
        model_name: 调用的ChatGPTH或者其他大模型的名称，参见openai_api_llm.py
            当前支持：gpt-3.5-turbo-1106_azure, gpt-4-1106-preview_azure, gpt-3.5-turbo-1106_api2d, gpt-4-1106-preview_api2d, gpt-3.5-turbo-16k-0613_api2d, gpt-3.5-turbo-16k_api2d, 
            gpt-3.5-turbo-0613_api2d, gpt-3.5-turbo_api2d, gpt-4-0613_api2d, gpt-4_api2d, qwen-72b-chat_hefei, qwen-14b-chat_hefei

        list_placeholder: 一个字符串，用于标识key_names_list当中需要进行列表循环操作的位置，例如对应于一个帖子的多个答案需要分别调用ChatGPT处理时需要设置该部分
        llm_output_key: 一个字符串，用于保存每次调用大模型处理时返回的文本在字典中保存的键
        do check: 是否对output_dir中存在的文件进行二次检查
    '''

    # 初始化模型接口
    if ( model_name.endswith("_api2d") or model_name.endswith("_hefei") or model_name.endswith("_azure") or model_name.endswith("_kimi")): # 参见openai_api_llm.py
            llm = OpenAI_LLM(model_name=model_name)
    else:
        raise ValueError("输入了非法的model_name，请设置该参数为gpt-3.5-turbo-1106_azure, gpt-4-1106-preview_azure, gpt-3.5-turbo-1106_api2d, gpt-4-1106-preview_api2d, gpt-3.5-turbo-16k-0613_api2d, gpt-3.5-turbo-16k_api2d, gpt-3.5-turbo-0613_api2d, gpt-3.5-turbo_api2d, gpt-4-0613_api2d, gpt-4_api2d, qwen-72b-chat_hefei, qwen-14b-chat_hefei, moonshot-v1-32k_kimi, moonshot-v1-128k_kimi, moonshot-v1-8k_kimi")
    
    # 检查key_names_list是否存在'list_placeholder'元素
    list_placeholder_i, list_placeholder_j = get_list_placeholder_index(key_names_list, list_placeholder=list_placeholder)
    # 设置内循环标志位
    if list_placeholder_i is not None:
        inner_loop_flag = True
    else:
        inner_loop_flag = False

    # 循环读取dict_list
    for key in range(len(dict_list)):
        temp_dict = dict_list[key]
        index_name_str = str(temp_dict[index_key_name]) # 可能为整数，需要转为字符串，方便判断和保存文件

        if temp_dict[index_key_name] not in files or do_check:
            if do_check:
                # 执行检查操作

                if os.path.exists(os.path.join(output_dir, index_name_str+'.json')):
                    try:
                        with open(os.path.join(output_dir, index_name_str+'.json'),'r',encoding='utf-8') as f :
                            temp_dict = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        print(os.path.join(output_dir, index_name_str+'.json'), " JSONDecodeError:", str(e))

            if inner_loop_flag:
                # 存在内循环时，需要逐一检查列表中的每个字典是否保存了大模型的处理结果
                inner_list = get_element_from_dict_with_index_list(temp_dict=temp_dict, index_list=key_names_list[list_placeholder_i][:list_placeholder_j])
                inner_loop_num = len(inner_list)

                for inner_loop_index in range(inner_loop_num):
                    if do_check:
                        if llm_output_key in inner_list[inner_loop_index]:
                            # 避免重复生成
                            continue

                    # 构建读取字典中的元素的传入列表
                    key_names = copy.deepcopy(key_names_list)
                    key_names[list_placeholder_i][list_placeholder_j] = inner_loop_index

                    total_prompt = prompt_from_dict_with_key_names(temp_dict=temp_dict, 
                                                                    general_prompt=general_prompt, 
                                                                    key_names=key_names)
                    message_text = [{"role": "user", "content": total_prompt}]
                    try:
                        # old_time放在程序运行开始的地方
                        old_time = time.time()
                        if model_name.endswith("_kimi"):
                            completion = llm.chat(
                                messages = message_text,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                stop=stop,
                                stream=stream,
                                add_system_prompt=add_system_prompt,
                                return_completions=return_completions)

                        elif num_tokens_from_messages(message_text) > 12000 and model_name.endswith("_azure"):
                            # 调用微软GPT4
                            completion = llm_azure_gpt4.chat(
                                messages = message_text,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                stop=stop,
                                stream=stream,
                                add_system_prompt=add_system_prompt,
                                return_completions=return_completions)
                        elif num_tokens_from_messages(message_text) > 12000:
                            # 调用API2D GPT-4
                            completion = llm_api2d_gpt4.chat(
                                messages = message_text,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                stop=stop,
                                stream=stream,
                                add_system_prompt=add_system_prompt,
                                return_completions=return_completions)
                        else:
                            # 调用model_name指定的LLM
                            completion = llm.chat(
                                messages = message_text,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                stop=stop,
                                stream=stream,
                                add_system_prompt=add_system_prompt,
                                return_completions=return_completions)

                        llm_answer = completion.choices[0].message.content
                        # 修改temp_dict
                        save_llm_answer_key_names = key_names[list_placeholder_i][:list_placeholder_j+1] + [llm_output_key]
                        add_value_to_dict_with_key_names(temp_dict, key_names= save_llm_answer_key_names, value=llm_answer)

                        with open(os.path.join(output_dir, index_name_str+'.json'), "w", encoding="utf-8") as f:
                            json.dump(temp_dict, f, indent=4, ensure_ascii=False)

                        index_str = "][".join([str(i) for i in key_names[list_placeholder_i][:list_placeholder_j+1]])
                        print(index_name_str+'.json'+ " 更新temp_dict[{}]成功".format(index_str))
                        current_time = time.time()
                        if (current_time - old_time) < 20:
                            time.sleep(45)
                        elif (current_time - old_time) < 30:
                            time.sleep(35)
                        elif (current_time - old_time) < 40:
                            time.sleep(25)
                        elif (current_time - old_time) < 50:
                            time.sleep(10)

                    except openai.APIError as e:
                        #Handle API error here, e.g. retry or log
                        print(f"【错误返回】OpenAI API returned an API Error: {e}")
                        # 错误码说明：https://openai.xiniushu.com/docs/guides/error-codes
                        # 错误码400
                        if e.code == 'content_filter':
                            '''
                            OpenAI API returned an API Error: Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}}
                            '''
                            # 样本可能触发安全过滤
                            if num_tokens_from_messages(message_text) > 12000:
                                completion = llm_api2d_gpt4.chat(
                                                messages = message_text,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=stop,
                                                stream=stream,
                                                add_system_prompt=add_system_prompt,
                                                return_completions=return_completions)
                            else:
                                completion = llm_api2d_gpt35.chat(
                                                messages = message_text,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=stop,
                                                stream=stream,
                                                add_system_prompt=add_system_prompt,
                                                return_completions=return_completions)
                                
                            llm_answer = completion.choices[0].message.content
                            # 修改temp_dict
                            save_llm_answer_key_names = key_names[list_placeholder_i][:list_placeholder_j+1] + [llm_output_key]
                            add_value_to_dict_with_key_names(temp_dict, key_names= save_llm_answer_key_names, value=llm_answer)
                            with open(os.path.join(output_dir, index_name_str+'.json'), "w", encoding="utf-8") as f:
                                json.dump(temp_dict, f, indent=4, ensure_ascii=False)

                            index_str = "][".join([str(i) for i in key_names[list_placeholder_i][:list_placeholder_j+1]])
                            print(index_name_str+'.json'+ " 更新temp_dict[{}]成功".format(index_str))
                            time.sleep(60)
                            pass

                        elif e.code == '429':
                            time.sleep(120)
                            pass

                        else:
                            # 样本可能触发安全过滤
                            if num_tokens_from_messages(message_text) > 12000:
                                completion = llm_api2d_gpt4.chat(
                                                messages = message_text,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=stop,
                                                stream=stream,
                                                add_system_prompt=add_system_prompt,
                                                return_completions=return_completions)
                            else:
                                completion = llm_api2d_gpt35.chat(
                                                messages = message_text,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=stop,
                                                stream=stream,
                                                add_system_prompt=add_system_prompt,
                                                return_completions=return_completions)
                                
                            llm_answer = completion.choices[0].message.content
                            # 修改temp_dict
                            save_llm_answer_key_names = key_names[list_placeholder_i][:list_placeholder_j+1] + [llm_output_key]
                            add_value_to_dict_with_key_names(temp_dict, key_names= save_llm_answer_key_names, value=llm_answer)
                            with open(os.path.join(output_dir, index_name_str+'.json'), "w", encoding="utf-8") as f:
                                json.dump(temp_dict, f, indent=4, ensure_ascii=False)

                            index_str = "][".join([str(i) for i in key_names[list_placeholder_i][:list_placeholder_j+1]])
                            print(index_name_str+'.json'+ " 更新temp_dict[{}]成功".format(index_str))
                            pass
                            
                    except openai.APIConnectionError as e:
                        #Handle connection error here
                        print(f"【错误返回】Failed to connect to OpenAI API: {e}")
                        # 本进程暂停60秒再访问
                        time.sleep(60)
                        pass
                    except openai.RateLimitError as e:
                        #Handle rate limit error (we recommend using exponential backoff)
                        print(f"【错误返回】OpenAI API request exceeded rate limit: {e}")
                        # 本进程暂停120秒再访问
                        time.sleep(120)
                        pass

            else:
                # 没有内循环
                if do_check:
                    if llm_output_key in temp_dict:
                        # 避免重复生成
                        continue
                # 构建读取字典中的元素的传入列表
                key_names = copy.deepcopy(key_names_list)
                total_prompt = prompt_from_dict_with_key_names(temp_dict=temp_dict, 
                                                                general_prompt=general_prompt, 
                                                                key_names=key_names)
                message_text = [{"role": "user", "content": total_prompt}]

                try:
                    # old_time放在程序运行开始的地方
                    old_time = time.time()
                    if num_tokens_from_messages(message_text) > 12000 and model_name.endswith("_azure"):
                        # 调用微软GPT4
                        completion = llm_azure_gpt4.chat(
                            messages = message_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            stream=stream,
                            add_system_prompt=add_system_prompt,
                            return_completions=return_completions)

                    elif num_tokens_from_messages(message_text) > 12000:
                        # 调用API2D GPT-4
                        completion = llm_api2d_gpt4.chat(
                            messages = message_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            stream=stream,
                            add_system_prompt=add_system_prompt,
                            return_completions=return_completions)
                    else:
                        # 调用model_name指定的LLM
                        completion = llm.chat(
                            messages = message_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            stop=stop,
                            stream=stream,
                            add_system_prompt=add_system_prompt,
                            return_completions=return_completions)

                    llm_answer = completion.choices[0].message.content
                    # 修改temp_dict
                    temp_dict[llm_output_key] = llm_answer
                    with open(os.path.join(output_dir, index_name_str+'.json'), "w", encoding="utf-8") as f:
                        json.dump(temp_dict, f, indent=4, ensure_ascii=False)

                    print(index_name_str+'.json'+ " 更新temp_dict成功")
                    current_time = time.time()
                    if (current_time - old_time) < 20:
                        time.sleep(45)
                    elif (current_time - old_time) < 30:
                        time.sleep(35)
                    elif (current_time - old_time) < 40:
                        time.sleep(25)
                    elif (current_time - old_time) < 50:
                        time.sleep(10)

                except openai.APIError as e:
                    #Handle API error here, e.g. retry or log
                    print(f"【错误返回】OpenAI API returned an API Error: {e}")
                    # 错误码说明：https://openai.xiniushu.com/docs/guides/error-codes
                    # 错误码400
                    if e.code == 'content_filter':
                        '''
                        OpenAI API returned an API Error: Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}}
                        '''
                        # 样本可能触发安全过滤

                        if num_tokens_from_messages(message_text) > 12000:
                            completion = llm_api2d_gpt4.chat(
                                            messages = message_text,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            top_p=top_p,
                                            frequency_penalty=frequency_penalty,
                                            presence_penalty=presence_penalty,
                                            stop=stop,
                                            stream=stream,
                                            add_system_prompt=add_system_prompt,
                                            return_completions=return_completions)
                        else:
                            completion = llm_api2d_gpt35.chat(
                                            messages = message_text,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            top_p=top_p,
                                            frequency_penalty=frequency_penalty,
                                            presence_penalty=presence_penalty,
                                            stop=stop,
                                            stream=stream,
                                            add_system_prompt=add_system_prompt,
                                            return_completions=return_completions)
                            
                        llm_answer = completion.choices[0].message.content
                        # 修改temp_dict
                        temp_dict[llm_output_key] = llm_answer

                        with open(os.path.join(output_dir, index_name_str+'.json'), "w", encoding="utf-8") as f:
                            json.dump(temp_dict, f, indent=4, ensure_ascii=False)

                        print(index_name_str+'.json'+ " 更新temp_dict成功")
                        time.sleep(60)
                        pass
                        
                    elif e.code == '429':
                        '''
                        OpenAI API returned an API Error: Error code: 429 - {'error': {'code': '429', 'message': 'Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2024-02-15-preview have exceeded token rate limit of your current OpenAI S0 pricing tier. Please retry after 1 second. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.'}}

                        '''
                        time.sleep(120)
                        pass
                    else:
                        time.sleep(60)
                        pass
                        
                except openai.APIConnectionError as e:
                    #Handle connection error here
                    print(f"【错误返回】Failed to connect to OpenAI API: {e}")
                    # 本进程暂停60秒再访问
                    time.sleep(60)
                    pass
                except openai.RateLimitError as e:
                    #Handle rate limit error (we recommend using exponential backoff)
                    print(f"【错误返回】OpenAI API request exceeded rate limit: {e}")
                    # 本进程暂停120秒再访问
                    time.sleep(120)
                    pass


class MultiProcessingLLM(multiprocessing.Process):
    '''
    功能：多进程调用LLM的基础类
    
    '''
    def __init__(self, 
                 idx_process, 
                 dict_list, 
                 key_names_list,
                 general_prompt,
                 files, 
                 num_process=32, # 总的进程数
                 index_key_name='id',
                 output_dir = './output_dir', 
                 model_name='gpt-4-1106-preview_api2d', 
                 list_placeholder = 'list_placeholder',
                 llm_output_key = 'chatgpt_output', # 保存大模型输出的对应的键
                 do_check=False,
                 temperature=0.7, # 大模型生成文本相关参数，下同
                 max_tokens=4096,
                 top_p=0.95,
                 frequency_penalty=0,
                 presence_penalty=0,
                 stop=None,
                 stream=False, # 禁止修改stream设置
                 add_system_prompt=True,
                 return_completions=True):
        '''
        功能：循环处理dict_list,进行调用ChatGPT生成回复

        输入格式:
            idx_process: int类型，进程号，用于识别和区分进程
            dict_list: 字典列表，参考格式：
                [
                    {
                        'id': 'xxx',
                        'name':  <str>,
                        'question': <str>,
                        'answer_list':
                            [
                                {
                                    'content': <str>,
                                },
                                ...
                            ]
                    }. # 第一个样本
                    ...
                
                ]

            key_names_list: 一个二维列表，元素个数与general_prompt的format占位符数目一致
                其中，字符串为'list_placeholder'的位置代表该处需要进行列表循环
                特别地，最多只允许一处'list_placeholder'
                [
                    ['questionTitle'], # 读取temp_dict['questionTitle']
                    ['questionDetail'], # 读取temp_dict['questionDetail']
                    ['answerOptions', 'list_placeholder', 'answerUserContent'] # 读取temp_dict['answerOptions'][i]['answerUserContent']
                ]
            general_prompt: 一个字符串，包含若干个format占位符,例如：
                '问：{}\n答：{}，请润色'
            files: 传入的已经保存在output_dir的文件名称汇总列表（去掉后缀）
            num_process: 总的进程数，这个是为了给多个账号分配不同的账号进行处理时使用，如果只有一个账号，则忽略即可！
            index_key_name:dict_list当中每个样本的标识id的对应键，通常为'id'
            output_dir: 处理完的样本输出的目录
            model_name: 调用的ChatGPTH或者其他大模型的名称，参见openai_api_llm.py
                当前支持：gpt-3.5-turbo-1106_azure, gpt-4-1106-preview_azure, gpt-3.5-turbo-1106_api2d, gpt-4-1106-preview_api2d, gpt-3.5-turbo-16k-0613_api2d, gpt-3.5-turbo-16k_api2d, 
                gpt-3.5-turbo-0613_api2d, gpt-3.5-turbo_api2d, gpt-4-0613_api2d, gpt-4_api2d, qwen-72b-chat_hefei, qwen-14b-chat_hefei

            list_placeholder: 一个字符串，用于标识key_names_list当中需要进行列表循环操作的位置，例如对应于一个帖子的多个答案需要分别调用ChatGPT处理时需要设置该部分
            llm_output_key: 一个字符串，用于保存每次调用大模型处理时返回的文本在字典中保存的键
            do check: 是否对output_dir中存在的文件进行二次检查
            temperature: 大模型生成时的温度系数
            ...

    '''
        multiprocessing.Process.__init__(self)
        self.idx_process = idx_process
        self.dict_list = dict_list
        self.files = files
        self.num_process = num_process
        self.general_prompt = general_prompt
        self.key_names_list = key_names_list

        self.index_key_name=index_key_name
        self.output_dir = output_dir
        self.model_name = model_name
        self.list_placeholder = list_placeholder
        self.llm_output_key = llm_output_key # 保存大模型输出的对应的键
        self.do_check = do_check

        self.temperature = temperature # 大模型生成文本相关参数，下同
        self.max_tokens = max_tokens
        self.top_p=top_p
        self.frequency_penalty=frequency_penalty
        self.presence_penalty=presence_penalty
        self.stop=stop
        self.stream=stream # 禁止修改stream设置
        self.add_system_prompt = add_system_prompt
        self.return_completions = return_completions

    def run(self):
        # 启动进程
        print(f'【提示】进程{self.idx_process}/{self.num_process}启动...' + str(datetime.datetime.now()))

        get_answer_use_chatgpt(dict_list=self.dict_list, 
                               key_names_list=self.key_names_list,
                               general_prompt=self.general_prompt,
                               files=self.files, 
                               index_key_name=self.index_key_name,
                               output_dir = self.output_dir, 
                               model_name=self.model_name, 
                               list_placeholder = self.list_placeholder,
                               llm_output_key = self.llm_output_key, # 保存大模型输出的对应的键
                               do_check=self.do_check,
                               temperature=self.temperature, # 大模型生成文本相关参数，下同
                               max_tokens=self.max_tokens,
                               top_p=self.top_p,
                               frequency_penalty=self.frequency_penalty,
                               presence_penalty=self.presence_penalty,
                               stop=self.stop,
                               stream=self.stream, # 禁止修改stream设置
                               add_system_prompt=self.add_system_prompt,
                               return_completions=self.return_completions)
        

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_process", type=int, default=10, help="The number of the process")
    parser.add_argument("-lp", '--use_load_raw_json_data_with_process', action='store_true', help="设置为True时，调用load_raw_json_data_with_process处理数据，这可能涉及一些个人的个性化操作，需要修改load_raw_json_data_with_process函数适配预处理过程")
    parser.add_argument("-c", "--do_check", action='store_true', help="对于调用Azure OpenAI服务处理后的每个结果进行检查（主要检查是否属于安全警告的样本以及输入长度过长的样本），从而重新调用其他模型解决")
    parser.add_argument("-m", "--model_name", type=str, default="gpt-4-1106-preview_api2d", help="默认 GPT-4（API2D）。可选：gpt-4-1106-preview_api2d, gpt-4-1106-preview_azure, gpt-3.5-turbo-1106_api2d, gpt-3.5-turbo-1106_azure, gpt-4_api2d, qwen-72b-chat_hefei, moonshot-v1-32k_kimi 等")
    
    parser.add_argument("-pc", "--prompt_config_path", type=str, default='./prompt_config.json', help="保存Prompt配置文件")
    
    parser.add_argument("-ij", "--input_json_data_path", type=str, default='./data/<input_json_data_path>.json', help="输入的待调用ChatGPT进行处理的样本数据集文件，文件格式为json")
    parser.add_argument("-it", "--input_txt_data_dir", type=str, default='<input_txt_data_dir>', help="输入的待调用ChatGPT进行处理的样本数据集文件目录，文件格式为txt")
    parser.add_argument("-o", "--output_json_dir", type=str, default='./data/<output_json_dir>', help="保存经过调用ChatGPT进行处理的样本的路径")

    parser.add_argument("-l", "--list_placeholder", type=str, default='list_placeholder', help="一个字符串，用于标识key_names_list当中需要进行列表循环操作的位置，例如对应于一个帖子的多个答案需要分别调用ChatGPT处理时需要设置该部分")
    parser.add_argument("-k", "--llm_output_key", type=str, default='chatgpt_answer', help="一个字符串，用于保存每次调用大模型处理时返回的文本在字典中保存的键")
    parser.add_argument("-id", "--index_key_name", type=str, default='id', help="dict_list当中每个样本的标识id的对应键，通常为'id'")

    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="大模型生成的温度系数")
    parser.add_argument("-mt", "--max_tokens", type=int, default=4096, help="大模型生成的输出允许的最大tokens数目")
    parser.add_argument("-tp", "--top_p", type=float, default=0.95, help="大模型生成的top_p采样系数")

    parser.add_argument("-fp", "--frequency_penalty", type=float, default=0, help="大模型生成的frequency_penalty系数")
    parser.add_argument("-pp", "--presence_penalty", type=float, default=0, help="大模型生成的presence_penalty系数")

    parser.add_argument("-s", "--add_system_prompt", action='store_true', help="是否添加system_prompt")

    args = parser.parse_args()

    model_name = args.model_name # 模型名称，参考openai_api_llm.py
    num_process = args.num_process # 运行进程数
    use_load_raw_json_data_with_process = args.use_load_raw_json_data_with_process
    do_check = args.do_check # 对于调用Azure OpenAI服务处理后的每个结果进行检查（主要检查是否属于安全警告的样本以及输入长度过长的样本），从而重新调用其他模型解决
    input_json_data_path = args.input_json_data_path
    input_txt_data_dir = args.input_txt_data_dir
    output_dir = args.output_json_dir

    list_placeholder = args.list_placeholder
    llm_output_key = args.llm_output_key
    index_key_name = args.index_key_name

    temperature = args.temperature
    max_tokens = args.max_tokens
    top_p = args.top_p
    frequency_penalty = args.frequency_penalty
    presence_penalty = args.presence_penalty

    add_system_prompt = args.add_system_prompt

    # 读取key_names_list与general_prompt
    # prompt_config.json

    with open(args.prompt_config_path,'r',encoding='utf-8') as f :
        prompt_config_dict = json.load(f)

    general_prompt = prompt_config_dict['general_prompt']
    key_names_list = prompt_config_dict['key_names_list']


    # 检查key_names_list是否存在'list_placeholder'元素
    list_placeholder_i, list_placeholder_j = get_list_placeholder_index(key_names_list, list_placeholder=list_placeholder)
    # 设置内循环标志位
    if list_placeholder_i is not None:
        inner_loop_flag = True
    else:
        inner_loop_flag = False


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取已经转换的文件名
    files = os.listdir(output_dir)
    files = [os.path.splitext(file)[0] for file in files]

    # 读取json数据或者txt目录数据
    if use_load_raw_json_data_with_process:
        if os.path.exists(input_json_data_path):
            raw_json_data = load_raw_json_data_with_process(input_json_data_path)
            print('【提示】通过load_raw_json_data_with_process读取raw_json_data完成！')
        else:
            print('【报错】--input_json_data_path指定的输入数据路径有误！')

    else:

        if os.path.exists(input_json_data_path):
            raw_json_data = load_raw_json_data(input_json_data_path)
            print('【提示】通过load_raw_json_data读取raw_json_data完成！')

        elif os.path.exists(input_txt_data_dir):
            raw_json_data = load_raw_txt_data(input_txt_data_dir)
            print('【提示】通过load_raw_txt_data读取raw_json_data完成！')

        else:
            print('【报错】输入数据路径有误！')



    total_dict_list = raw_json_data['data']

    # 过滤掉已经转换的
    rawlist = total_dict_list # 任务列表
    old_rawlist = total_dict_list # 任务列表
    
    if do_check:
        rawlist1 = [example for example in old_rawlist if str(example[index_key_name]) not in files]
        rawlist2 = []
        # 检查所有已经处理的样本
        for file in files:
            try:
                with open(os.path.join(output_dir, file+'.json'),'r',encoding='utf-8') as f :
                    temp_dict = json.load(f)

                if inner_loop_flag == False:
                    if llm_output_key not in temp_dict:
                        rawlist2.append(temp_dict)
                else:
                    # 循环检查
                    # 存在内循环时，需要逐一检查列表中的每个字典是否保存了大模型的处理结果
                    inner_list = get_element_from_dict_with_index_list(temp_dict=temp_dict, index_list=key_names_list[list_placeholder_i][:list_placeholder_j])
                    inner_loop_num = len(inner_list)
                    for inner_loop_index in range(inner_loop_num):
                        if llm_output_key not in inner_list[inner_loop_index]:
                            rawlist2.append(temp_dict)
                            break

            except json.decoder.JSONDecodeError as e:
                print(os.path.join(output_dir, file+'.json'), " JSONDecodeError:", str(e))
                os.remove(os.path.join(output_dir, file+'.json'))
                rawlist1 = rawlist1 + [example for example in old_rawlist if str(example[index_key_name])==file]

        rawlist = rawlist2 + rawlist1

    else:
        # 忽略已处理列表
        rawlist = [example for example in old_rawlist if str(example[index_key_name]) not in files]

    print('【提示】检查需要处理的样本已完成！')
    print("待处理数据个数：", len(rawlist))

    # 一、执行ChatGPT对多个回复进行润色
    if len(rawlist) > 0:
        # 存在样本还没有经过ChatGPT进行润色了
        if len(rawlist) < num_process:
            # 对于样本数量小于进程数的，重新设置进程数
            num_process = len(rawlist)

        # 根据进程数切割任务
        temp=[]
        id_list = []
        step=len(rawlist)//num_process
        for i in range(0,num_process-1):
            temp.append(rawlist[step*i:step*i+step])
            id_list.append([ str(example[index_key_name]) for example in rawlist[step*i:step*i+step] ])
        temp.append(rawlist[step*(num_process-1):])
        id_list.append([ str(example[index_key_name]) for example in rawlist[step*(num_process-1):] ])
        #print(temp)
        
        # 初始化进程

        print("【启动】初始化进程...")

        processes = []
        for i in range(0, num_process):
            try:
                # 重新读取已经转换的文件名
                files = os.listdir(output_dir)
                files = [os.path.splitext(file)[0] for file in files]

                process = MultiProcessingLLM(
                                        idx_process=i, 
                                        dict_list=temp[i], 
                                        key_names_list=key_names_list,
                                        general_prompt=general_prompt,
                                        files=files, 
                                        num_process=num_process, # 总的进程数
                                        index_key_name=index_key_name,
                                        output_dir=output_dir, 
                                        model_name=model_name, 
                                        list_placeholder=list_placeholder,
                                        llm_output_key=llm_output_key, # 保存大模型输出的对应的键
                                        do_check=do_check,
                                        temperature=temperature, # 大模型生成文本相关参数，下同
                                        max_tokens=max_tokens,
                                        top_p=top_p,
                                        frequency_penalty=frequency_penalty,
                                        presence_penalty=presence_penalty,
                                        stop=None,
                                        stream=False, # 禁止修改stream设置
                                        add_system_prompt=add_system_prompt,
                                        return_completions=True)
                process.start()
                processes.append(process)
            except Exception as e:
                print('启动【进程{i}】出错，5秒后重试...')
                time.sleep(5)
                continue

        while len(files) < len(old_rawlist) or do_check:
            if len(processes) == 0:
                print('所有进程均完成任务！')
                break

            for i, p in enumerate(processes):
                if not p.is_alive():
                    processes.remove(p)
                    print('-------------------------------------------')
                    print(f"【进程{i}】正在重启...")
                    time.sleep(30)
                    # 重新读取已经转换的文件名
                    files = os.listdir(output_dir)
                    files = [os.path.splitext(file)[0] for file in files]
                    # 重新启动进程
                    if (set(id_list[i]) < set(files)) and (not do_check):
                        # 列表元素已经全部转换，且不执行do_check，不重启进程
                        print(f"【进程{i}】已经完成任务，不再重启...")
                    else:
                        if do_check and len(temp[i]) > 0:
                            # 更新temp[i]
                            new_temp_i = []
                            for example in temp[i]:
                                # 执行检查进行确认
                                if os.path.exists(os.path.join(output_dir, example[index_key_name]+'.json')):
                                    try:
                                        with open(os.path.join(output_dir, example[index_key_name]+'.json'),'r',encoding='utf-8') as f :
                                            temp_dict = json.load(f)

                                        if inner_loop_flag == False:
                                            if llm_output_key not in temp_dict:
                                                new_temp_i.append(temp_dict)
                                        else:
                                            # 循环检查
                                            # 存在内循环时，需要逐一检查列表中的每个字典是否保存了大模型的处理结果
                                            inner_list = get_element_from_dict_with_index_list(temp_dict=temp_dict, index_list=key_names_list[list_placeholder_i][:list_placeholder_j])
                                            inner_loop_num = len(inner_list)
                                            for inner_loop_index in range(inner_loop_num):
                                                if llm_output_key not in inner_list[inner_loop_index]:
                                                    new_temp_i.append(temp_dict)
                                                    break

                                    except json.decoder.JSONDecodeError as e:
                                        print(os.path.join(output_dir, file+'.json'), " JSONDecodeError:", str(e))
                                        new_temp_i.append(example)

                                else:
                                    # 文件不存在
                                    new_temp_i.append(example)
                            
                            if len(new_temp_i) > 0:
                                temp[i] = new_temp_i # 更新temp[i]
                                process = MultiProcessingLLM(
                                                        idx_process=i, 
                                                        dict_list=temp[i], 
                                                        key_names_list=key_names_list,
                                                        general_prompt=general_prompt,
                                                        files=files, 
                                                        num_process=num_process, # 总的进程数
                                                        index_key_name=index_key_name,
                                                        output_dir=output_dir, 
                                                        model_name=model_name, 
                                                        list_placeholder=list_placeholder,
                                                        llm_output_key=llm_output_key, # 保存大模型输出的对应的键
                                                        do_check=do_check,
                                                        temperature=temperature, # 大模型生成文本相关参数，下同
                                                        max_tokens=max_tokens,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=None,
                                                        stream=False, # 禁止修改stream设置
                                                        add_system_prompt=True,
                                                        return_completions=True)
                                process.start()
                                processes.append(process)
                            else:
                                temp[i] = []
                                print(f"【进程{i}】已经完成任务，不再重启...")
                        
                        elif not do_check and len(temp[i]) > 0:
                            # 更新temp[i]
                            new_temp_i = []
                            for example in temp[i]:
                                # 执行检查进行确认
                                if os.path.exists(os.path.join(output_dir, example[index_key_name]+'.json')):
                                    continue

                                else:
                                    # 文件不存在
                                    new_temp_i.append(example)
                            
                            if len(new_temp_i) > 0:
                                temp[i] = new_temp_i # 更新temp[i]
                                process = MultiProcessingLLM(
                                                        idx_process=i, 
                                                        dict_list=temp[i], 
                                                        key_names_list=key_names_list,
                                                        general_prompt=general_prompt,
                                                        files=files, 
                                                        num_process=num_process, # 总的进程数
                                                        index_key_name=index_key_name,
                                                        output_dir=output_dir, 
                                                        model_name=model_name, 
                                                        list_placeholder=list_placeholder,
                                                        llm_output_key=llm_output_key, # 保存大模型输出的对应的键
                                                        do_check=do_check,
                                                        temperature=temperature, # 大模型生成文本相关参数，下同
                                                        max_tokens=max_tokens,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=None,
                                                        stream=False, # 禁止修改stream设置
                                                        add_system_prompt=add_system_prompt,
                                                        return_completions=True)
                                process.start()
                                processes.append(process)
                            else:
                                temp[i] = []
                                print(f"【进程{i}】已经完成任务，不再重启...")

                        
                        else:
                            if len(processes) == 0:
                                print('【提示】所有进程均完成任务，不再重启进程！')
                                break

                            print(f"【进程{i}】已经完成任务，不再重启...")
                            continue
        
        print('【提示】“执行ChatGPT对多个回复进行排序”任务结束！')



if __name__ == "__main__":

    try:
        start = time.time()
        main()
        end = time.time()
        print("运行程序花费了%s秒" % (end - start))
    except Exception as e:
        print(e)

        