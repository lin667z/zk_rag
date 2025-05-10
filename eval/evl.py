import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch


class AnswerGenerator:
    def __init__(self, lora_checkpoint_path):
        self.device = "cuda"
        self.config = PeftConfig.from_pretrained(lora_checkpoint_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name_or_path,
            trust_remote_code=True,
            pad_token="<|im_end|>"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_checkpoint_path,
            device_map=self.device,
        )
        self.model.eval()

    def generate(self, system_prompt, user_input):
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=10240,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
            ** inputs,
            do_sample = True,
            max_new_tokens = 512,
            temperature = 0.4,
            top_p = 0.85,
            repetition_penalty = 1.2,
            )

            # 提取新生成的内容
            input_length = inputs.input_ids.size(1)
            generated_ids = outputs[:, input_length:]
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # 清理可能的结束标记
            return answer.split("<|im_end|>")[0].strip()

def process_questions(input_path, output_path):
    # 初始化生成器
    generator = AnswerGenerator(
        lora_checkpoint_path="../output/checkpoint-73102"
    )

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            try:
                data = json.loads(line.strip())
                answer = generator.generate(
                    system_prompt=data["system_prompt"],
                    user_input=data["input"]
                )
                json.dump({"answer": answer}, fout, ensure_ascii=False)
                fout.write("\n")
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {str(e)}")


if __name__ == "__main__":
    process_questions(
        input_path="../dataset/test.jsonl",
        output_path="output_answers.jsonl"
    )
