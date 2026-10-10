



conda activate llama_factory
cd .

# txt转换为json
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_sentence' --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/en' --output_json_dir '../pipeline_data/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk

# 根据json文件转换为向量
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' --store_top_path '../pipeline_data/vector_stores/zh_sentence' 
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --store_top_path '../pipeline_data/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/en_json' --store_top_path '../pipeline_data/vector_stores/en'

# 对json文件进行检索
# 中文数据集------检索仅使用"question"
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/test.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"


# 中文数据集------检索使用"question_with_options"
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/test.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"


# 英文数据集------检索仅使用"question"
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
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
    --embedding_model_name="stella-base-en-v2" \
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


# 英文数据集------检索仅使用"question_with_options"
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"



python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"


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
    --query_key_name="question_with_options"




# 生成问题与答案之间的解析
# 中文
cd .
python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95


python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95



python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/dev/retrive_use_question_with_options/stella-base-zh-v2/dev.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/dev/retrive_use_question_with_options/stella-base-zh-v2/dev_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95

# 英文
cd .
python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain_en.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95


python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain_en.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/US/test/retrive_use_question_with_options/stella-base-en-v2/test.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/test/retrive_use_question_with_options/stella-base-en-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95



python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain_en.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/US/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/dev/retrive_use_question_with_options/stella-base-en-v2/dev_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95




# 构建标准训练集
# 数据路径：../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi
# 构建训练集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_train" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset

# 构建测试集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../pipeline_data/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_test" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset




