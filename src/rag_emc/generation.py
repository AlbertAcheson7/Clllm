from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Any, Mapping

class Generator:
    """
    处理实体类型验证的生成部分。
    默认模型路径: model/meta-llama/Llama-3.1-8B-Instruct
    """
    def __init__(self, model_path="model/meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 尝试加载模型，处理flash attention不可用的情况
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            print("✅ 使用 FlashAttention2 加载模型")
        except (ImportError, ValueError) as e:
            print(f"⚠️ FlashAttention2 不可用，使用标准attention: {e}")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✅ 使用标准attention加载模型")
            except Exception as e2:
                print(f"❌ 模型加载失败，尝试CPU模式: {e2}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print("✅ 使用CPU模式加载模型")
        
        self.model.eval()

    def validate(self, mention: str, retrieval_results: dict):
        validation_judgments = {}
        for source, retrieved_data in retrieval_results.items():
            if not retrieved_data or not retrieved_data.get('document'):
                validation_judgments[source] = {
                    "prompt": "N/A",
                    "answer": "No document"
                }
                continue
            context = retrieved_data.get('document', 'Not found')
            user_prompt_for_source = (
                "You are a biomedical expert system. Your task is to determine whether the given 'mention' refers to a biomedical named entity.\n"
                "Use the background information to make your decision.\n\n"
                f"Background Information:\n{context}\n\n"
                f"Mention:\n{mention}\n\n"
                "Answer with one of the following options (choose only one): [No, Chemical, Disease]."
            )
            messages = [
                {"role": "user", "content": user_prompt_for_source}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_len = inputs['input_ids'].shape[1]
            terminators = [self.tokenizer.eos_token_id]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_tokens = outputs[0, input_len:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            validation_judgments[source] = {
                'prompt': prompt,
                'answer': answer
            }
        result = {
            'mention': mention,
            'retrieval_results': retrieval_results,
            'validation_judgments': validation_judgments
        }
        return result
