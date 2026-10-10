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

# File Name: generate_question_with_knowledges.py
# Description: 多进程进行RAG检索

# Reference: 
# 

# 使用示例

'''
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/test.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"


'''



import os
import re
import torch
import json
import time
import datetime
import argparse
from tqdm import tqdm
import multiprocessing

from retriever import SearchResult, OneStageRetriever
from embedding import load_huggingface_embedding


def load_retriever(store_path, embedding_name, device):
    embedding_model = load_huggingface_embedding(name=embedding_name, device=device)
    one_stage_retriever = OneStageRetriever(store_path=store_path, embedding_model=embedding_model)
    #two_stage_retriever = TwoStageRetriever(embedding_model=embedding_model)
    return one_stage_retriever #, two_stage_retriever


def load_jsonl_data(data_path):
    '''
    功能：从data_path中读取jsonl文件，并且返回列表
    '''

    # 读取所有行并解析为JSON对象的列表
    with open(data_path, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line) for line in file]

    return data_list


def convert_questions_dict_example_to_new_example(questions_dict):
    '''
    questions_dict = {'question': '经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（\u3000\u3000）。', 
                      'options': {'A': '2小时', 'B': '4小时内', 'C': '8小时内', 'D': '12小时内', 'E': '24小时内'}, 
                      'answer': '24小时内', 
                      'meta_info': '卫生法规', 
                      'answer_idx': 'E'}
    
    '''

    options_str = '\n'.join([f'{k}. {v}' for k, v in questions_dict["options"].items()])

    answer_with_idx = questions_dict['answer_idx'] + '. ' + questions_dict['answer']

    question_with_options = questions_dict['question'] + '\n' + options_str

    return {
        'question_with_options': question_with_options,
        'answer_with_idx': answer_with_idx,
        'question': questions_dict['question'],
        'options': questions_dict['options'],
        'answer': questions_dict['answer'],
        'meta_info': questions_dict['meta_info'], 
        'answer_idx': questions_dict['answer_idx']
    }


def generate_new_data_list(data_list):

    new_data_list = []
    i = 0
    for questions_dict in data_list:
        new_questions_dict = convert_questions_dict_example_to_new_example(questions_dict)
        new_questions_dict['id'] = i
        new_data_list.append(new_questions_dict)
        i = i + 1

    return new_data_list


def run_generate_question_with_knowledges(
        idx_process,
        data_list,
        onestageretriever,
        device,
        embedding_name,
        output_dir,
        topk_knowledge=10,
        knowledge_threshold=0.65,
        query_key_name="question"):
    

    retrieve_results = []

    print(f"【提示】进程{idx_process}启动")
    for example in data_list:
        query = example[query_key_name]
        search_result = onestageretriever.search(
                            query=query,
                            topk_knowledge=topk_knowledge,
                            knowledge_threshold=knowledge_threshold,
                        )
        
        retrieve_info = {
            "retrieve_query": query,
            "embedding_name": embedding_name,
            "topk_knowledge": topk_knowledge,
            "knowledge_threshold": knowledge_threshold,
            "knowledges_info": {
                "knowledges": search_result.knowledges,
                "scores": search_result.knowledge_scores,
                "knowledge_names": search_result.knowledge_names,
            },
            "retrieval_time": search_result.time,
        }

        example['retrieve_info'] = retrieve_info

        retrieve_results.append(example)

    print(f"【提示】进程{idx_process}保存结果到{output_dir}/{idx_process}.json")
    with open(os.path.join(output_dir, str(idx_process)+'.json'), "w", encoding="utf-8") as f:
        json.dump(retrieve_results, f, indent=4, ensure_ascii=False)

    
class MultiProcessingRetriever(multiprocessing.Process):
    '''
    功能：多进程调用检索模型进行知识检索
    
    '''

    def __init__(self, 
                 idx_process,
                 num_process,
                 data_list,
                 onestageretriever,
                 device,
                 embedding_name,
                 output_dir,
                 topk_knowledge=10,
                 knowledge_threshold=0.65,
                 query_key_name="question"):
        '''
        '''
        multiprocessing.Process.__init__(self)
        self.idx_process = idx_process
        self.num_process = num_process # 总的进程数
        self.data_list = data_list
        self.onestageretriever = onestageretriever
        self.device = device
        self.embedding_name = embedding_name
        self.output_dir = output_dir
        self.topk_knowledge = topk_knowledge
        self.knowledge_threshold = knowledge_threshold
        self.query_key_name = query_key_name

    def run(self):
        # 启动进程
        print(f'【提示】进程{self.idx_process}/{self.num_process}启动...' + str(datetime.datetime.now()))

        run_generate_question_with_knowledges(
            idx_process=self.idx_process,
            data_list=self.data_list,
            onestageretriever=self.onestageretriever,
            device=self.device,
            embedding_name=self.embedding_name,
            output_dir=self.output_dir,
            topk_knowledge=self.topk_knowledge,
            knowledge_threshold=self.knowledge_threshold,
            query_key_name=self.query_key_name)


def main():
    #torch.multiprocessing.set_start_method('spawn')
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_process", type=int, default=10, help="The number of the process")
    parser.add_argument("-id", "--index_key_name", type=str, default='id', help="dict_list当中每个样本的标识id的对应键，通常为'id'")
    parser.add_argument("-tp", "--topk_knowledge", type=int, default=10, help="检索时返回的知识数量")
    parser.add_argument("-kt", "--knowledge_threshold", type=float, default=0.65, help="检索时的知识相似度阈值")
    parser.add_argument("-i", "--input_jsonl_path", type=str, default='../pipeline_data/questions/Mainland/train.jsonl', help="json格式的知识文档的目录")
    parser.add_argument("-e", "--embedding_model_name", type=str, default='stella-base-zh-v2', help="字符串形式，输入的embedding模型的名称")
    parser.add_argument("-v", "--store_path", type=str, default='../pipeline_data/vector_stores/zh_paragraph', help="张量数据库保存的目录，其子目录类似于：stella-base-zh-v2/0_whole")
    parser.add_argument("-od", "--output_json_dir", type=str, default='../pipeline_data/questions_with_knowledge/zh_paragraph', help="张量保存的目录")
    parser.add_argument("-of", "--output_json_file", type=str, default='train.json', help="张量保存的目录")
    parser.add_argument("-d", "--device_ids", type=str, default='0,1,2,3,4,5,6,7', help="字符串形式，允许使用的device_id，以,分隔")
    parser.add_argument("-qk", "--query_key_name", type=str, default='question', help="字符串形式，进行检索时的字典键，为'question_with_options'或者'question'")
    args = parser.parse_args()

    num_process = args.num_process
    index_key_name = args.index_key_name
    query_key_name = args.query_key_name
    topk_knowledge = args.topk_knowledge
    knowledge_threshold = args.knowledge_threshold
    input_jsonl_path = args.input_jsonl_path
    embedding_model_name = args.embedding_model_name
    store_path = args.store_path
    output_json_dir = args.output_json_dir
    output_dir = os.path.join(output_json_dir, 'retrive_use_'+query_key_name, embedding_model_name, 'multiprocessingsave')
    output_json_file = args.output_json_file
    # 
    output_json_file_path = os.path.join(output_json_dir, 'retrive_use_'+query_key_name, embedding_model_name, output_json_file)


    device_ids = args.device_ids

    device_ids = [int(i) for i in device_ids.split(',')] # '0,1,2,3,4,5,6,7' --> [0, 1, 2, 3, 4, 5, 6, 7]
    device_ids = ['cuda:'+str(i) for i in device_ids]
    device_num = len(device_ids)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取数据
    data_list = load_jsonl_data(data_path=input_jsonl_path)
    # 数据预处理
    rawlist = generate_new_data_list(data_list)

    print("样本数量：", len(rawlist))
    print("样本示例：", rawlist[0])

    # 切割数据
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

    # 加载模型
    onestageretriever_lists = []
    device_lists = []
    for i in range(num_process):
        device = device_ids[i%device_num]
        device_lists.append(device)
        onestageretriever = load_retriever(store_path=store_path, embedding_name=embedding_model_name, device=device)
        onestageretriever_lists.append(onestageretriever)

    print("onestageretriever_lists=", onestageretriever_lists)

    print("【启动】初始化进程...")

    processes = []
    for i in range(0, num_process):
        try:
            process =MultiProcessingRetriever(
                                        idx_process=i,
                                        num_process=num_process,
                                        data_list=temp[i],
                                        onestageretriever=onestageretriever_lists[i],
                                        device=device_lists[i],
                                        embedding_name=embedding_model_name,
                                        output_dir=output_dir,
                                        topk_knowledge=topk_knowledge,
                                        knowledge_threshold=knowledge_threshold,
                                        query_key_name=query_key_name)
            
            process.start()
            processes.append(process)
        except Exception as e:
            print(f'启动【进程{i}】出错，5秒后重试...错误信息：{e}')
            time.sleep(5)
            continue


    while len(os.listdir(output_dir)) < num_process:
        for i, p in enumerate(processes):
            if len(processes) == 0:
                print('【提示】所有进程均完成任务，不再重启进程！')
                break
            if not p.is_alive():
                processes.remove(p)
                if not os.path.exists(os.path.join(output_dir, str(i)+'.json')):
                    print('-------------------------------------------')
                    print(f"【进程{i}】正在重启...")
                    time.sleep(30)

                    process =MultiProcessingRetriever(
                                            idx_process=i,
                                            num_process=num_process,
                                            data_list=temp[i],
                                            onestageretriever=onestageretriever_lists[i],
                                            device=device_lists[i],
                                            embedding_name=embedding_model_name,
                                            output_dir=output_dir,
                                            topk_knowledge=topk_knowledge,
                                            knowledge_threshold=knowledge_threshold)
                
                    process.start()
                    processes.append(process)
                
                else:
                    temp[i] = []
                    print(f"【进程{i}】已经完成任务，不再重启...")

    if len(os.listdir(output_dir)) == num_process:
        # 合并文件
        total_data = []
        file_names = os.listdir(output_dir)
        for file_name in file_names:
            json_path = os.path.join(output_dir, file_name)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_data = total_data + data

        # 排序
        total_data = sorted(total_data, key=lambda x : x["id"])

        with open(output_json_file_path, "w", encoding="utf-8") as f:
            json.dump(total_data, f, indent=4, ensure_ascii=False)

        print("保存成功：", output_json_file_path)


if __name__ == "__main__":

    try:
        start = time.time()
        main()
        end = time.time()
        print("运行程序花费了%s秒" % (end - start))
    except Exception as e:
        print(e)