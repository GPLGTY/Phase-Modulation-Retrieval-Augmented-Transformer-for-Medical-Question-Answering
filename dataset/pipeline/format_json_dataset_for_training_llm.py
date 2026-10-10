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


# Author: Yirong Chen <eeyirongchen@mail.scut.edu.cn or 1132461715@qq.com>, Wenjing Han
# Date: 2024.05.24

# File Name: format_json_dataset_for_training_llm.py
# Description: 
# ../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi


# 使用示例
'''
conda activate llama_factory
cd dataset/pipeline

# 构建训练集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_train" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset

python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_train_500(example)" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --sample_num=500


# 构建测试集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_test" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset


# 构建测试集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_test_300" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset \
    --sample_num=300
    


# 构建无RAG的训练集与测试集

## 构建训练集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="MedQA_Mainland_train" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --no_retrival \
    --do_register_dataset

## 构建测试集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="MedQA_Mainland_test" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --no_retrival \
    --do_register_dataset

    
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../RAG_MedQA_Mainland_test_300.json" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="MedQA_Mainland_test_300" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --no_retrival \
    --do_register_dataset



'''

import os
import random
import re
import copy
import time
import json
import argparse
from tqdm import tqdm


SYSTEM_PROMPT = '''你是一位专业的医生，掌握大量医学专业知识及临床决策能力，对相关医学概念有深刻的理解，掌握与健康和疾病相关的生物医学、心理学和社会学原理，并能用于指导学习和临床实践。掌握人体正常结构、功能、生理过程，以及常见病及重要疾病的临床特征、发病机制和诊治原则。你能准确地回答各类医学或非医学问题，并给出详细的分析过程。'''

MEDQA_QUESTION_PROMPT ='''给定以下从参考书籍查询到的**医学知识**，请回答相关**医学问题**。其中，知识可能与问题有关，也可能与问题无关，你需要判断是否需要利用提供的医学知识来回答问题。请给出解析与答案。
#医学知识#
{}

#医学问题#
{}

#回答格式#
解析：（详细的分析过程）
答案：编号选项

请严格按照**回答格式**回答上面的医学问题。
'''


NO_RETRIVAL_MEDQA_QUESTION_PROMPT ='''请回答下面的问题。
#医学问题#
{}

#回答格式#
解析：（详细的分析过程）
答案：编号选项

请严格按照**回答格式**回答上面的医学问题。
'''





MEDQA_ANSWER_PROMPT = '''针对你提的问题，我的解析过程与回答如下：\n\n{}\n\n因此，针对上述问题的答案为：\n{}'''



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


def load_raw_json_data(data_path):
    '''
    功能：读取'./data/total_raw_data.json'并返回完整的数据raw_json_data
    返回：
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
        if os.path.isdir(data_path):
            # 判断输入的路径是否为目录
            print('【提示】正在从data_path={}指定的目录读取原始数据...'.format(data_path))
            raw_json_data = []
            json_paths = [os.path.join(data_path, file_name) for file_name in get_filenames(data_path, ignore_start_with_string='~', end_with_string="json", sorted_by_num=True)]
            for json_path in json_paths:
                with open(json_path,'r',encoding='utf-8') as f :
                    example_json_data = json.load(f)

                raw_json_data.append(example_json_data)
            
            print('原始数据集总共的样本（求助贴）数=', len(raw_json_data))
            raw_json_data = {'data': raw_json_data}
            return raw_json_data

        else:

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



def generate_openai_format_dataset(json_data,
                                   system_prompt=SYSTEM_PROMPT,
                                   user_prompt_key_names_list= [
                                            ['knowledges_for_llm'],
                                            ['question_with_options']     
                                    ],
                                    user_prompt=MEDQA_QUESTION_PROMPT,
                                    assistant_prompt_key_names_list= [
                                            ['chatgpt_explain'],
                                            ['answer_with_idx']     
                                    ],
                                    assistant_prompt=MEDQA_ANSWER_PROMPT,
                                    do_test=False):
    '''
    功能：构建**openai** 格式的数据集
    do_test=True时，不添加"role": "assistant"

    output_json_data = [
        {
            "messages": [
            {
                "role": "system",
                "content": "系统提示词（选填）"
            },
            {
                "role": "user",
                "content": "用户指令"
            },
            {
                "role": "assistant",
                "content": "模型回答"
            }
            ],
            "id": xx,
            ...
        }
    ]
    
    '''

    output_json_data = []
    for example in json_data:
        messages = []
        # 添加系统提示词
        messages.append(
            {
                "role": "system",
                "content": system_prompt
            }
        )
        # 添加用户指令
        messages.append(
            {
                "role": "user",
                "content": prompt_from_dict_with_key_names(temp_dict=example, general_prompt=user_prompt, key_names=user_prompt_key_names_list)
            }
        )
        # 添加助手回复
        if not do_test:
            messages.append(
                {
                    "role": "assistant",
                    "content": prompt_from_dict_with_key_names(temp_dict=example, general_prompt=assistant_prompt, key_names=assistant_prompt_key_names_list)
                }
            )

        output_json_data.append(
            {
                "id": example["id"],
                "meta_info": example["meta_info"],
                "question_with_options": example["question_with_options"],
                "answer_with_idx": example["answer_with_idx"],
                "answer_idx": example["answer_idx"],
                "messages": messages,
                "retrieve_info": example["retrieve_info"]
            }
        )

    return output_json_data



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json_data_path", type=str, default='../data/<input_csv_data_path>.json', help="输入的json文件或者存储批量json文件的目录")
    parser.add_argument("-df", "--dataset_info_path", type=str, default='../data/dataset_info.json', help="注册标准数据集的文件路径")
    parser.add_argument("-n", "--dataset_name", type=str, default='rag_medqa_train', help="注册标准数据集名称")
    parser.add_argument("-rd", "--dataset_info_relative_dir", type=str, default='MedQA/', help="保存的数据文件相对于--dataset_info_path的路径，以斜杆结尾。例如--dataset_info_path=data/dataset_info.json；保存的文件为data/MedQA/rag_medqa_train.json则--dataset_info_relative_dir=MedQA/")
    parser.add_argument("-o", "--output_json_data_dir", type=str, default='../data/<output_json_data_dir>', help="输出的json格式数据的保存目录")
    parser.add_argument("-r", "--do_register_dataset", action='store_true', help="是否注册数据集信息到--dataset_info_path指定的dataset_info.json文件")
    parser.add_argument("-nr", "--no_retrival", action='store_true', help="设置为True时，不将检索知识加到对话当中")
    parser.add_argument("-t", "--do_test", action='store_true', help="是否构建测试集，设置为True时执行构建测试集的模式")
    parser.add_argument("-s", "--sample_num", type=int, default=None, help="对数据进行采样的采样个数")
    args = parser.parse_args()

    input_json_data_path=args.input_json_data_path
    dataset_info_path=args.dataset_info_path
    dataset_name=args.dataset_name
    dataset_info_relative_dir = args.dataset_info_relative_dir
    dataset_file_name = dataset_name + ".json"
    output_json_data_dir=args.output_json_data_dir
    do_register_dataset=args.do_register_dataset
    no_retrival=args.no_retrival
    sample_num=args.sample_num

    if not os.path.exists(output_json_data_dir):
        os.makedirs(output_json_data_dir)


    dataset_info_example_dict = {
        "file_name": "",
        "formatting": "sharegpt",
        "columns": {
        "messages": "messages"
        },
        "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
        }
    }

    if do_register_dataset:
        # 注册数据集信息

        if os.path.exists(dataset_info_path):
            # 存在dataset_info文件，则读取
            with open(dataset_info_path,'r',encoding='utf-8') as f :
                dataset_info = json.load(f)
        else:
            dataset_info = {}

        # 判断是否已经存在注册名
        if dataset_name in dataset_info:
            raise ValueError("【错误】提供的--dataset_name={}已经存在于{}，请确保dataset_name命名与-- dataset_info_path中的键名不冲突。如果只考虑更新数据集文件，运行本文件时请移除--do_register_dataset，这样将不进行注册操作".format(dataset_name, dataset_info_path))
        
    raw_json_data = load_raw_json_data(data_path=input_json_data_path)

    if no_retrival:
        openai_format_dataset = generate_openai_format_dataset(json_data=raw_json_data["data"],
                                                            system_prompt=SYSTEM_PROMPT,
                                                            user_prompt_key_names_list= [
                                                                            ['question_with_options']],
                                                            user_prompt=NO_RETRIVAL_MEDQA_QUESTION_PROMPT,
                                                            assistant_prompt_key_names_list= [
                                                                            ['chatgpt_explain'],
                                                                            ['answer_with_idx']],
                                                            assistant_prompt=MEDQA_ANSWER_PROMPT)

    else:

        openai_format_dataset = generate_openai_format_dataset(json_data=raw_json_data["data"],
                                                            system_prompt=SYSTEM_PROMPT,
                                                            user_prompt_key_names_list= [
                                                                            ['knowledges_for_llm'],
                                                                            ['question_with_options']],
                                                            user_prompt=MEDQA_QUESTION_PROMPT,
                                                            assistant_prompt_key_names_list= [
                                                                            ['chatgpt_explain'],
                                                                            ['answer_with_idx']],
                                                            assistant_prompt=MEDQA_ANSWER_PROMPT)
    
    # 保存数据集
    # 判断是否采样：
    if sample_num is not None:
        if sample_num > 0 and sample_num < len(openai_format_dataset):
            openai_format_dataset = random.sample(openai_format_dataset, sample_num)

    # 保存json数据
    output_json_data_path = os.path.join(output_json_data_dir, dataset_file_name)
    with open(output_json_data_path, "w", encoding="utf-8") as f:
        json.dump(openai_format_dataset, f, indent=4, ensure_ascii=False)

    print("成功保存openai_format_dataset数据到{}".format(output_json_data_path))

    if do_register_dataset:
        dataset_info_example = copy.deepcopy(dataset_info_example_dict)
        dataset_info_example["file_name"] = dataset_info_relative_dir + dataset_file_name
        dataset_info[dataset_name] = dataset_info_example

        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=4, ensure_ascii=False)

        print("成功保存注册数据集信息到{}".format(dataset_info_path))
        print("注册的数据集的命名：", dataset_name)
        print("注册的数据集信息为：", dataset_info_example)





if __name__ == "__main__":

    try:
        start = time.time()
        main()
        end = time.time()
        print("运行程序花费了%s秒" % (end - start))
    except Exception as e:
        print(e)




