import re
import json
import os
import argparse
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter # 辅助拆分

def paragraphs_fusion(paragraphs: list[str], do_chunk=False, max_knowledge_len=1600, language_type='chinese') -> list[str]:
    # 陈艺荣添加
    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ". ", "?", "!", ""],
            chunk_size=max_knowledge_len,
            chunk_overlap=0,
            length_function=len,
        )
    
    result = []
    for j, p in enumerate(paragraphs):
        if language_type=='chinese' or language_type=='zh':
            for en, zh in zip([",", "?", "!", ";", ":"], ["，", "？", "！", "；", "："]):
                p.replace(en, zh)
        elif language_type=='english' or language_type=='en':
            for en, zh in zip([",", "?", "!", ";", ":"], ["，", "？", "！", "；", "："]):
                p.replace(zh, en)
        
        # 当前范围的首个正文段落无条件append
        if j == 0:
            if do_chunk and len(p) > max_knowledge_len:
                # 增加文本长度约束
                splits = text_splitter.split_text(text=p)
                result.extend(splits)
            else:
                result.append(p)

            continue
        # result中最后添加的段落结束了，则append当前正文段落
        if result[-1][-1] in [".", "。", "！", "？", "?", "!"]:
            if do_chunk and len(p) > max_knowledge_len:
                # 增加文本长度约束
                splits = text_splitter.split_text(text=p)
                result.extend(splits)
            else:
                result.append(p)
        # result中最后添加的段落未结束，则将当前正文段落拼接到末尾
        else:
            if do_chunk and len(p) > max_knowledge_len:
                # 增加文本长度约束
                splits = text_splitter.split_text(text=p)
                result.extend(splits)

            else:
                p = "\n" + p
                result[-1] += p

    return result


def txt_to_json(txt_path: str, json_path: str, do_chunk=False, max_knowledge_len=1600, language_type='chinese', min_knowledge_len=10):
    # 陈艺荣添加
    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", ". ", "?", "!", ";"],
            chunk_size=max_knowledge_len,
            chunk_overlap=0,
            length_function=len,
        )


    with open(txt_path, "r", encoding="utf-8-sig") as f:
        paragraphs = [p.strip() for p in f if p.strip()]

    title_prefix_pattern = r"^#{1,}"

    # 找到标题段落，记录标题段落id和标题级别
    title_paragraph_ids_levels = []
    for i, p in enumerate(paragraphs):
        if title_prefix_matched := re.match(title_prefix_pattern, p):
            title_level = len(title_prefix_matched.group())
            title_paragraph_ids_levels.append((i, title_level))

    knowledges = []
    titles = []
    title_levels = []

    if len(title_paragraph_ids_levels) == 0:
        # 没有标题符号
        for i, p in enumerate(paragraphs):
            if do_chunk and len(p) > max_knowledge_len:
                splits = text_splitter.split_text(text=p)
                knowledges.extend(splits)
            else:
                knowledges.append(p)

    else:
        for i in range(-1, len(title_paragraph_ids_levels)):
            # 获取需要处理的正文段落范围 start:end
            if i == -1:
                start = 0
                end = (
                    title_paragraph_ids_levels[i + 1][0]
                    if i + 1 < len(title_paragraph_ids_levels)
                    else len(paragraphs)
                )
                knowledges.extend(paragraphs_fusion(paragraphs[start:end], do_chunk=do_chunk, max_knowledge_len=max_knowledge_len, language_type=language_type))

            else:
                if i == len(title_paragraph_ids_levels) - 1:
                    start = title_paragraph_ids_levels[i][0] + 1
                    end = len(paragraphs)
                else:
                    start = title_paragraph_ids_levels[i][0] + 1
                    end = title_paragraph_ids_levels[i + 1][0]

                try:
                    update_level_index = title_levels.index(
                        title_paragraph_ids_levels[i][1]
                    )
                    titles[update_level_index:] = [
                        paragraphs[title_paragraph_ids_levels[i][0]]
                    ]
                    title_levels[update_level_index:] = [title_paragraph_ids_levels[i][1]]
                except:
                    titles.append(paragraphs[title_paragraph_ids_levels[i][0]])
                    title_levels.append(title_paragraph_ids_levels[i][1])

                if start == end:
                    knowledges.append("\n".join(titles))
                else:
                    knowledges.extend(
                        list(
                            map(
                                lambda x: "\n".join(titles + [x]),
                                paragraphs_fusion(paragraphs[start:end], do_chunk=do_chunk, max_knowledge_len=max_knowledge_len, language_type=language_type),
                            )
                        )
                    )

    # 对knowledges进行过滤，去掉无用信息
    knowledges = [k for k in knowledges if len(k) > min_knowledge_len]


    with open(json_path, "w") as f:
        json.dump(
            {"knowledges": {i: k for i, k in enumerate(knowledges)}},
            fp=f,
            indent=4,
            ensure_ascii=False,
        )


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



def build_jsons(txt_dir: str, json_dir: str, do_chunk=False, max_knowledge_len=1600, language_type='chinese', min_knowledge_len=10):
    txt_paths = [os.path.join(txt_dir, file_name) for file_name in get_filenames(txt_dir, ignore_start_with_string='~', end_with_string="txt", sorted_by_num=False)]

    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    json_paths = [
        os.path.join(json_dir, os.path.basename(path).split(".")[0] + ".json")
        for path in txt_paths
    ]
    for txt_path, json_path in tqdm(zip(txt_paths, json_paths), total=len(txt_paths)):
        txt_to_json(txt_path=txt_path, json_path=json_path, do_chunk=do_chunk, max_knowledge_len=max_knowledge_len, language_type=language_type, min_knowledge_len=min_knowledge_len)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--input_txt_dir", type=str, default='../pipeline_data/textbooks/zh_paragraph', help="txt格式的知识文档的目录")
    parser.add_argument("-oj", "--output_json_dir", type=str, default='../pipeline_data/textbooks/zh_paragraph_json', help="json格式的知识文档的目录")
    parser.add_argument("-l", "--language_type", type=str, default='chinese', help="知识库的语言类型，当前提供chinese、english两种选择")
    parser.add_argument("-maxkl", "--max_knowledge_len", type=int, default=1800, help="知识的最大长度")
    parser.add_argument("-minkl", "--min_knowledge_len", type=int, default=5, help="知识的最小长度")
    parser.add_argument("-c", '--do_chunk', action='store_true', help="设置为True时，对输入的知识进行拆分时会判断长度是否大于max_knowledge_len，对于大于max_knowledge_len的样本会进行二次拆分")

    args = parser.parse_args()

    input_txt_dir = args.input_txt_dir
    output_json_dir = args.output_json_dir
    language_type = args.language_type
    max_knowledge_len = args.max_knowledge_len
    min_knowledge_len = args.min_knowledge_len
    do_chunk = args.do_chunk

    if not os.path.exists(output_json_dir):
        os.mkdir(output_json_dir)

    build_jsons(txt_dir=input_txt_dir, 
                json_dir=output_json_dir, 
                do_chunk=args.do_chunk, 
                max_knowledge_len=args.max_knowledge_len, 
                language_type=args.language_type, 
                min_knowledge_len=min_knowledge_len)
