from openai import OpenAI
import json
import os
from datetime import datetime

# conda activae py38bert

def ner_demo():
    """使用GPT-3.5-turbo进行命名实体识别的demo"""
    client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-iOH4N6zieNOI61bNEYtfWI6a6qQa5MqyHj7C4yLqKUaOsHv9',
    )

    # 读取JSON文件
    input_file_path = '/home/zhangzy/CLllm/data/data_pre/data_pred/BC5CDR/merged_bc5cdr_test_data.json'
    
    print("=" * 50)
    print("NER Demo - 使用GPT-3.5-turbo进行命名实体识别")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误：输入文件 {input_file_path} 不存在")
        return None
    
    # 读取JSON数据
    print(f"正在读取文件: {input_file_path}")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共找到 {len(data)} 条数据")
    
    # 准备保存结果
    results = []
    
    # 处理每条数据
    for i, item in enumerate(data):
        test_sentence = item['text']
        sentence_id = item.get('sentence_id', f'unknown_{i}')
        
        print(f"\n正在处理 {i+1}/{len(data)}: {sentence_id}")
        print(f"句子: {test_sentence[:100]}{'...' if len(test_sentence) > 100 else ''}")
        
        # 构建NER任务的prompt
        ner_prompt = f"""
        "Please extract all possible entity mentions related to either medicine or chemistry from the following sentence.\n"
        "Output the result in this exact JSON format:\n"
        "{{"
        "  \"Medical\": [\"entity_name_1\", \"entity_name_2\"],"
        "  \"Chemical\": [\"entity_name_1\", \"entity_name_2\"]"
        "}}"
        "If there are no entities of a type, return an empty list for that type.\n"
        "Only output valid JSON. No explanation.\n"
        "Sentence: {test_sentence}"
        """
        ner_prompt_allpossible = f""" "Please extract all possible entity mentions related to either medicine or chemistry from the following sentence.\n"
        "Include not only obvious entities, but also borderline cases, abbreviations, synonyms, or partial mentions that could potentially refer to a medical or chemical concept.\n"
        "Even if you are uncertain, include the entity.\n"
        "Output the result in this exact JSON format:\n"
        "{{"
        "  \"Medical\": [\"entity_name_1\", \"entity_name_2\"],\n"
        "  \"Chemical\": [\"entity_name_1\", \"entity_name_2\"]\n"
        "}}"
        "If there are no entities of a type, return an empty list for that type.\n"
        "Only output valid JSON. No explanation.\n"
        "Sentence: {test_sentence}"
        """

        try:
            # 调用GPT-3.5-turbo进行NER识别
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": ner_prompt_allpossible,
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=0.1,  # 降低温度以获得更稳定的结果
            )
            
            # 获取识别结果
            ner_result = chat_completion.choices[0].message.content
            print(f"识别结果: {ner_result}")
            
            # 尝试解析JSON格式的结果
            try:
                parsed_result = json.loads(ner_result)
            except json.JSONDecodeError:
                print(f"警告：结果不是有效的JSON格式，保存原始文本")
                parsed_result = {"raw_result": ner_result}
            
            # 保存结果
            result_item = {
                "sentence_id": sentence_id,
                "original_text": test_sentence,
                "gpt_ner_result": parsed_result,
                "raw_response": ner_result,
                "original_entities": item.get('entities', [])  # 保存原始标注（如果有）
            }
            results.append(result_item)
            
        except Exception as e:
            print(f"处理句子时出错: {e}")
            # 保存错误信息
            result_item = {
                "sentence_id": sentence_id,
                "original_text": test_sentence,
                "error": str(e),
                "original_entities": item.get('entities', [])
            }
            results.append(result_item)
    
    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ner_results_{timestamp}.json"
    
    print(f"\n保存结果到文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理 {len(results)} 条数据")
    print(f"结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    # 运行NER demo
    results = ner_demo()
    
    if results:
        print("\n" + "=" * 50)
        print("处理统计:")
        print("=" * 50)
        
        successful_count = len([r for r in results if 'error' not in r])
        error_count = len([r for r in results if 'error' in r])
        
        print(f"成功处理: {successful_count} 条")
        print(f"处理失败: {error_count} 条")
        
        # 显示前几个成功的结果示例
        if successful_count > 0:
            print("\n前3个成功结果示例:")
            print("-" * 30)
            count = 0
            for result in results:
                if 'error' not in result and count < 3:
                    print(f"句子ID: {result['sentence_id']}")
                    print(f"原文: {result['original_text'][:100]}{'...' if len(result['original_text']) > 100 else ''}")
                    print(f"GPT识别结果: {result['gpt_ner_result']}")
                    print("-" * 30)
                    count += 1











"""
      Successfully uninstalled typing_extensions-4.8.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
wandb 0.16.4 requires PyYAML, which is not installed.
streamlit 1.32.0 requires blinker<2,>=1.0.0, which is not installed.
simpletransformers 0.70.0 requires datasets, which is not installed.
simpletransformers 0.70.0 requires scipy, which is not installed.
simpletransformers 0.70.0 requires tokenizers, which is not installed.
simpletransformers 0.70.0 requires transformers>=4.31.0, which is not installed.
altair 5.2.0 requires jsonschema>=3.0, which is not installed.
tensorflow-gpu 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 4.25.3 which is incompatible.
tensorflow-gpu 2.11.0 requires tensorboard<2.12,>=2.11, but you have tensorboard 2.14.0 which is incompatible.
Successfully installed annotated-types-0.7.0 anyio-4.5.2 distro-1.9.0 exceptiongroup-1.3.0 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 jiter-0.9.1 openai-1.95.1 pydantic-2.10.6 pydantic-core-2.27.2 sniffio-1.3.1 tqdm-4.67.1 typing-extensions-4.13.2
"""