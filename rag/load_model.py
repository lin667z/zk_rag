from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from rag.format_prompt import prompt_sys
from rag.db_store import VectorDatabase

def local_model(prompt, query):
    # 设置设备（自动检测GPU）
    device = "cuda"

    # 1. 加载配置
    lora_checkpoint_path = "../output/checkpoint-73102"  # 替换为实际路径
    config = PeftConfig.from_pretrained(lora_checkpoint_path)

    # 2. 加载基础模型
    base_model_name = config.base_model_name_or_path
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,  # 适用于Qwen等特殊模型
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        pad_token="<|im_end|>"  # 特殊token处理[9](@ref)
    )
    # 3. 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_path,
        device_map=device,
    )
    # 处理没有pad token的情况
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # 设置为评估模式

    # 4. prompt
    formatted_prompt = f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    # 5. 推理示例
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=10240,
        truncation=True,
        add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(  # 或使用model
            **inputs,
            do_sample = True,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.2,  # 防止重复条款
            # eos_token_id=tokenizer.im_end_id  # 终止符设置
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False,clean_up_tokenization_tokens=True)

    return full_response

# 4. 权重合并
# 如果希望永久保存合并后的模型（用于部署）
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("merged_model")
# tokenizer.save_pretrained("merged_model")


# 测试推理
if __name__ == "__main__":
    # 初始化数据库
    vdb = VectorDatabase(
            index_path="cache/faiss_index.bin",
            metadata_path="cache/metadata.pkl"
        )
    vdb.load()
    query = ""
    references = vdb.search(query=query)
    contents = []
    for result in references:
        contents.append(result["text"])

    prompt = prompt_sys(contents)
    print(local_model(prompt=query, query=query))