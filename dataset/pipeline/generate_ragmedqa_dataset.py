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

# File Name: generate_ragmedqa_dataset.py
# Description: 

# Reference: 
# 

# 使用示例


import os
import json
import time
import datetime
import argparse


RAG_PROMPT_TEMPLATE = """
#目标#
你是一个专业的医生，请基于以下从知识库当中检索到的知识回答问题。如果无法从知识中得到答案，忽略已知知识直接回答问题。你的回答风格需要参考回答格式，先生成分析和推理答案的解析过程，然后提供回答的答案，也就是"解析：...\n答案：..."的模式。

#参考的回答案例#
<问题示例>
用户输入的问题：下列哪项是耳、肾毒性最大的氨基糖苷类抗生素？（　　）\nA. 庆大霉素\nB. 卡那霉素\nC. 西索米星\nD. 奈替米星\nE. 新霉素
回答：
解析：新霉素是一种氨基糖苷类抗生素，具有广泛的抗菌谱，但因其耳毒性和肾毒性较大，临床应用受到限制。新霉素的耳毒性主要表现为听力下降、耳鸣和平衡障碍，而肾毒性则表现为肾小管损伤和肾功能减退。因此，在临床使用时需要严格掌握适应症和剂量，避免不良反应的发生。其他选项的氨基糖苷类抗生素虽然也具有一定的耳毒性和肾毒性，但相对较小。
答案：E. 新霉素
</问题示例>

#知识#
知识库中检索的知识如下所示：
<知识点>
{knowledge}
</知识点>

#问题#
请针对以下用户输入的选择题，结合检索到的知识给出准确的回答。
用户输入的问题：{query}


#回答格式#
解析：[详细的推理和分析过程]
答案：[答案选项，如果答案是多个则以换行符分隔]
"""


def load_data_with_explain_from_dir(data_dir):

    raw_json_data = []

    files = os.listdir(data_dir)

    for file in files:
        file_path = os.path.join(data_dir, file)
        with open(file_path,'r',encoding='utf-8') as f :
            example = json.load(f)

        raw_json_data.append()

        








def convert_questions_and_knowledge(question_str, 
                                    knowledge_lists,
                                    rag_prompt_template=RAG_PROMPT_TEMPLATE):

    knowledges_for_llm = "\n\n".join(knowledge_lists)
    content = rag_prompt_template.format(query=question_str, knowledge=knowledges_for_llm)

    return content









def standardize_json_data(input_json_data_path, 
                          output_json_data_path,
                          system_prompt=None):
    '''
    '''
    print('【提示】正在从data_path={}指定的文件读取原始数据...'.format(input_json_data_path))
    with open(input_json_data_path,'r',encoding='utf-8') as f :
        raw_json_data = json.load(f)

    new_data = []
    for example in raw_json_data:

        messages = []
        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        question_str = example["question_with_options"]
        knowledge_lists = example["retrieve_info"]["knowledges_info"]["knowledges"]
        user_content = convert_questions_and_knowledge(question_str=question_str, 
                                                       knowledge_lists=knowledge_lists,
                                                       rag_prompt_template=RAG_PROMPT_TEMPLATE)
        
        messages.append(
                {
                    "role": "user",
                    "content": user_content
                }
            )
        
        

        



        new_example_dict = {
            "id": example["id"]
        }





def main():
    # 主函数入口
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_base_dir", type=str, default='../pipeline_data/questions_with_knowledge', help="调用generate_question_with_knowledges.py生成的questions_with_knowledge的存储目录")
    parser.add_argument("-i", "--output_dataset_dir", type=str, default='../RAGMedQA', help="调用generate_question_with_knowledges.py生成的questions_with_knowledge的存储目录")
    args = parser.parse_args()

    input_base_dir = args.input_base_dir
    output_dataset_dir = args.output_dataset_dir

    nation_types = ['US', 'Mainland']
    spilt_types = ['train', 'dev', 'test']
    retrive_types = ['retrive_use_question', 'retrive_use_question_with_options']

    system_prompt = {
        'US': "You are a professional doctor with a doctoral degree. You have a solid foundation in medical knowledge, including anatomy, physiology, pathology, etc. You have passed the licensed physician qualification exam with full marks, and can accurately answer various medical questions and medical qualification exam questions in clinical practice, traditional Chinese medicine, dentistry, public health, etc., and provide detailed explanations.",
        "Mainland": "你是一个专业的医生，学历为博士，你掌握了扎实的医学基础知识，包括解剖学、生理学、病理学等，以满分的成绩通过了执业医师资格考试，能准确无误地回答临床、中医、口腔、公共卫生等各类医学问题和医学资格考试题并且能详细做出解释。"
    }



    # 完整的数据目录
    # input_base_dir/nation_types/spilt_types/retrive_types/{retrive_types}.json

    '''
    输出的数据格式
        [
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
        ]
    }
    ]
    
    '''













if __name__ == "__main__":

    try:
        start = time.time()
        main()
        end = time.time()
        print("运行程序花费了%s秒" % (end - start))
    except Exception as e:
        print(e)