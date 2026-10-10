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

# File Name: format_json_dataset_for_eval_from_jsonl.py
# Description: 韩文静毕设，基于大模型的大五人格分析


# 使用示例
'''
cd .

# 无检索
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="../pipeline_data/MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/predict_MedQA_Mainland_test_300_use_Qwen1.5-14B-Chat_with_temperature_0.7_top_p_0.8_top_k_20/generated_predictions.jsonl" \
    --output_xlsx_result_path="./results/predict_MedQA_Mainland_test_300_use_Qwen1.5-14B-Chat_with_temperature_0.7_top_p_0.8_top_k_20.xlsx"

# 带知识检索
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="../pipeline_data/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/predict_RAG_MedQA_Mainland_test_300_use_Qwen1.5-14B-Chat_with_temperature_0.7_top_p_0.8_top_k_20/generated_predictions.jsonl" \
    --output_xlsx_result_path="./results/predict_RAG_MedQA_Mainland_test_300_use_Qwen1.5-14B-Chat_with_temperature_0.7_top_p_0.8_top_k_20.xlsx"

# JiYiChat
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="../pipeline_data/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/predict_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-500_with_temperature_0.7_top_p_0.8_top_k_20/generated_predictions.jsonl" \
    --output_xlsx_result_path="./results/predict_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-500_with_temperature_0.7_top_p_0.8_top_k_20.xlsx"


# predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-1500_with_temperature_0.7_top_p_0.8_top_k_20
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="../pipeline_data/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-4500_with_temperature_0.7_top_p_0.8_top_k_20/generated_predictions.jsonl" \
    --output_xlsx_result_path="./results/predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-4500_with_temperature_0.7_top_p_0.8_top_k_20.xlsx"

    
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="../pipeline_data/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/" \
    --input_jsonl_subdir_startwiths="predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train_checkpoint" \
    --output_xlsx_result_path="./results/output_xlsx_result" \
    --output_xlsx_total_result_path="./results/output_xlsx_result/0_all_scores.xlsx"
    



'''


import os
import re
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm



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


def load_jsonl_data(data_path):
    '''
    功能：从data_path中读取jsonl文件，并且返回列表
    '''

    # 读取所有行并解析为JSON对象的列表
    with open(data_path, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line) for line in file]

    return data_list


def get_answer_idx_from_text(text):
    if "因此，针对上述问题的答案为：" in text:
        text = text.split("因此，针对上述问题的答案为：")[-1]
        if ". " in text:
            text = text.split(". ")[0]

        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)
    
    elif "答案：" in text:
        text = text.split("答案：")[-1]
        if ". " in text:
            text = text.split(". ")[0]

        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)
        
    elif "the answer to the above question is:" in text:
        text = text.split("the answer to the above question is:")[-1]
        if ". " in text:
            text = text.split(". ")[0]

        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)
        
    elif "the correct answer is " in text:
        text = text.split("the correct answer is ")[-1]
        if ". " in text:
            text = text.split(". ")[0]

        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)

    elif "the correct answer." in text:
        pattern = r"making option ([A-Z]) is the correct answer."
        match = re.search(pattern, text)

        pattern2 = r"which is why option ([A-Z]) is the correct answer."
        match2 = re.search(pattern2, text)

        pattern3 = r"making option ([A-Z]) the correct answer."
        match3 = re.search(pattern3, text)

        pattern4 = r"which is why option ([A-Z]) the correct answer."
        match4 = re.search(pattern4, text)
        if match:
            letter = match.group(1)
            print(f"获取到的字母是: {letter}")
            return letter
        elif match2:
            letter = match2.group(1)
            print(f"获取到的字母是: {letter}")
            return letter
        elif match3:
            letter = match3.group(1)
            print(f"获取到的字母是: {letter}")
            return letter
        elif match4:
            letter = match4.group(1)
            print(f"获取到的字母是: {letter}")
            return letter
        else:
            return "回答异常，可能需要手工核对"

    elif "The correct answer should be " in text:
        text = text.split("The correct answer should be ")[-1]
        if ". " in text:
            text = text.split(". ")[0]

        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)

    elif "which is Option " in text:
        text = text.split("which is Option ")[-1]
        if ". " in text:
            text = text.split(". ")[0]
        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)

    elif "which is option " in text:
        text = text.split("which is option ")[-1]
        if ". " in text:
            text = text.split(". ")[0]
        uppercase_letters = re.findall(r'[A-Z]', text)
        if len(uppercase_letters) == 1:
            return uppercase_letters[0]
        elif len(uppercase_letters) == 0:
            return "回答异常，可能需要手工核对"
        elif len(uppercase_letters) > 1:
            uppercase_letters = sorted(uppercase_letters)
            return str(uppercase_letters)

    else:
         return "回答异常，可能需要手工核对"




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json_data_path", type=str, default='../data/<input_json_data_path>.json', help="输入的json文件或者存储批量json文件的目录,提供target信息")
    parser.add_argument("-jl", "--input_jsonl_data_path", type=str, default='../data/<input_jsonl_data_path>.jsonl', help="输入的jsonl文件，提供predict信息")

    parser.add_argument("-sjd", "--input_jsonl_subdir_startwiths", type=str, default='predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train_checkpoint', help="约束input_llm_predict_adjectives_jsonl_dir的子目录，只读取子目录中以指定字符串开头的文件")
    parser.add_argument("-jfn", "--input_llm_predict_jsonl_file_name", type=str, default='generated_predictions.jsonl', help="指定读取input_jsonl_data_path目录下的子目录中的jsonl文件名称")

    parser.add_argument("-o", "--output_json_data_path", type=str, default='../data/<output_json_data_path>.json', help="输出的json结果文件")
    parser.add_argument("-x", "--output_xlsx_result_path", type=str, default='../data/<output_xlsx_result_path>.xlsx', help="保存批量统计时的实验结果(当--input_jsonl_data_path为目录时，该项也为目录)")
    parser.add_argument("-r", "--output_xlsx_total_result_path", type=str, default='../data/<output_xlsx_total_result_path>.xlsx', help="保存批量统计时的实验结果得分")
    args = parser.parse_args()
    input_json_data_path = args.input_json_data_path
    input_jsonl_data_path = args.input_jsonl_data_path
    output_json_data_path = args.output_json_data_path
    output_xlsx_result_path = args.output_xlsx_result_path
    output_xlsx_total_result_path = args.output_xlsx_total_result_path

    input_jsonl_subdir_startwiths = args.input_jsonl_subdir_startwiths
    input_llm_predict_jsonl_file_name = args.input_llm_predict_jsonl_file_name

    raw_json_data = load_raw_json_data(data_path=input_json_data_path)

    if os.path.exists(input_jsonl_data_path):

        # 对目录批量处理
        if os.path.isdir(input_jsonl_data_path):

            if not os.path.exists(output_xlsx_result_path):
                os.makedirs(output_xlsx_result_path)

            sub_dirs = os.listdir(input_jsonl_data_path)
            sub_dirs = [sub_dir for sub_dir in sub_dirs if sub_dir.startswith(input_jsonl_subdir_startwiths) and os.path.isdir(os.path.join(input_jsonl_data_path, sub_dir))]
            each_experiments_score = []

            for sub_dir in tqdm(sub_dirs):
                print("当前处理文件：{}".format(sub_dir))
                if os.path.exists(os.path.join(input_jsonl_data_path, sub_dir, input_llm_predict_jsonl_file_name)):
                    llm_predict_data = load_jsonl_data(data_path=os.path.join(input_jsonl_data_path, sub_dir, input_llm_predict_jsonl_file_name))

                    total_dict_list = []
                    total_dict_list_for_xlsx = []

                    if len(llm_predict_data) != len(raw_json_data["data"]):
                        raise ValueError("【错误】len(llm_predict_data) != len(raw_json_data[\"data\"])")

                    sample_num = len(llm_predict_data)

                    total_score = 0

                    for i in range(sample_num):
                        example = raw_json_data["data"][i]
                        predict_text = llm_predict_data[i]["predict"]
                        # 提取回答中的答案
                        predict_answer_idx = get_answer_idx_from_text(predict_text)
                        example["predict_answer_idx"] = predict_answer_idx
                        if example["predict_answer_idx"] == example["answer_idx"]:
                            example["predict_score"] = 1
                        else:
                            example["predict_score"] = 0

                        total_dict_list.append(example)
                        temp_example =  {
                            "id": example["id"],
                            "领域": example["meta_info"],
                            "问题": example["question_with_options"],
                            "正确答案": example["answer_idx"],
                            "预测答案": example["predict_answer_idx"],
                            "评分": example["predict_score"],
                            "LLM原始输出": predict_text
                        }

                        total_score = total_score + example["predict_score"]

                        for i in range(10):
                            if i < len(example["retrieve_info"]["knowledges_info"]["knowledges"]):
                                temp_example["检索知识{}知识库".format(i+1)] = example["retrieve_info"]["knowledges_info"]["knowledge_names"][i]
                                temp_example["检索知识{}".format(i+1)] = example["retrieve_info"]["knowledges_info"]["knowledges"][i]
                                temp_example["检索知识{}相似度".format(i+1)] = example["retrieve_info"]["knowledges_info"]["scores"][i]
                            else:
                                temp_example["检索知识{}知识库".format(i+1)] = ""
                                temp_example["检索知识{}".format(i+1)] = ""
                                temp_example["检索知识{}相似度".format(i+1)] = ""

                        total_dict_list_for_xlsx.append(temp_example)

                    # 保存结果
                    current_output_xlsx_result_path = os.path.join(output_xlsx_result_path, sub_dir+".xlsx")
                    df = pd.DataFrame(total_dict_list_for_xlsx)
                    df.to_excel(current_output_xlsx_result_path, index=False)

                    print("成功保存统计结果到{}".format(current_output_xlsx_result_path))

                    # 匹配checkpoint
                    checkpoint_pattern = r"checkpoint_(\d+\.\d+|\d+)"
                    checkpoint_match = re.search(checkpoint_pattern, sub_dir)
                    # 如果找到匹配项，打印出来
                    if checkpoint_match:
                        checkpoint_value = checkpoint_match.group(1)  # 提取匹配的数字部分
                        checkpoint = int(checkpoint_value)

                    else:
                        checkpoint = "未知"

                    # 匹配温度系数
                    temperature_pattern = r"temperature_(\d+\.\d+|\d+)"
                    temperature_match = re.search(temperature_pattern, sub_dir)
                    # 如果找到匹配项，打印出来
                    if temperature_match:
                        temperature_value = temperature_match.group(1)  # 提取匹配的数字部分
                        temperature = float(temperature_value)

                    else:
                        temperature = "未知"

                    # 匹配top_p
                    top_p_pattern = r"top_p_(\d+\.\d+|\d+)"
                    top_p_match = re.search(top_p_pattern, sub_dir)
                    # 如果找到匹配项，打印出来
                    if top_p_match:
                        top_p_value = top_p_match.group(1)  # 提取匹配的数字部分
                        top_p = float(top_p_value)

                    else:
                        top_p = "未知"



                    each_experiments_score.append(
                        {
                            "结果文件": sub_dir+".xlsx",
                            "checkpoint": checkpoint,
                            "temperature": temperature,
                            "top_p": top_p,
                            "score": total_score
                        }
                    )

            df = pd.DataFrame(each_experiments_score)
            df.to_excel(output_xlsx_total_result_path, index=False)

            print("成功保存统计结果到{}".format(output_xlsx_total_result_path))





        else:

            llm_predict_data = load_jsonl_data(data_path=input_jsonl_data_path)

            total_dict_list = []
            total_dict_list_for_xlsx = []

            if len(llm_predict_data) != len(raw_json_data["data"]):
                raise ValueError("【错误】len(llm_predict_data) != len(raw_json_data[\"data\"])")

            sample_num = len(llm_predict_data)

            for i in range(sample_num):
                example = raw_json_data["data"][i]
                predict_text = llm_predict_data[i]["predict"]
                # 提取回答中的答案
                predict_answer_idx = get_answer_idx_from_text(predict_text)
                example["predict_answer_idx"] = predict_answer_idx
                if example["predict_answer_idx"] == example["answer_idx"]:
                    example["predict_score"] = 1
                else:
                    example["predict_score"] = 0

                total_dict_list.append(example)
                temp_example =  {
                    "id": example["id"],
                    "领域": example["meta_info"],
                    "问题": example["question_with_options"],
                    "正确答案": example["answer_idx"],
                    "预测答案": example["predict_answer_idx"],
                    "评分": example["predict_score"],
                    "LLM原始输出": predict_text
                }

                for i in range(10):
                    if i < len(example["retrieve_info"]["knowledges_info"]["knowledges"]):
                        temp_example["检索知识{}知识库".format(i+1)] = example["retrieve_info"]["knowledges_info"]["knowledge_names"][i]
                        temp_example["检索知识{}".format(i+1)] = example["retrieve_info"]["knowledges_info"]["knowledges"][i]
                        temp_example["检索知识{}相似度".format(i+1)] = example["retrieve_info"]["knowledges_info"]["scores"][i]
                    else:
                        temp_example["检索知识{}知识库".format(i+1)] = ""
                        temp_example["检索知识{}".format(i+1)] = ""
                        temp_example["检索知识{}相似度".format(i+1)] = ""

                total_dict_list_for_xlsx.append(temp_example)

            # 保存结果
            df = pd.DataFrame(total_dict_list_for_xlsx)
            df.to_excel(output_xlsx_result_path, index=False)

            print("成功保存统计结果到{}".format(output_xlsx_result_path))








if __name__ == "__main__":

    try:
        start = time.time()
        main()
        end = time.time()
        print("运行程序花费了%s秒" % (end - start))
    except Exception as e:
        print(e)

