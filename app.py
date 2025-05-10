from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag.format_prompt import prompt_sys
from rag.db_store import VectorDatabase
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 初始化标志
_initialized = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = app.logger

# 全局变量存储模型和数据库
global_model = None
global_tokenizer = None
vdb = None
# 在全局变量部分新增
conversation_history = []
MAX_HISTORY_LENGTH = 20  # 最大保留历史轮次
MAX_TOKENS = 102400      # 模型最大token限制


def initialize_components():
    """初始化模型和数据库组件"""
    global global_model, global_tokenizer, vdb, _initialized

    # 确保只初始化一次
    if not _initialized:
        if global_model is None:
            # 1. 初始化数据库
            vdb = VectorDatabase(
                index_path="./rag/cache/faiss_index.bin",
                metadata_path="./rag/cache/metadata.pkl"
            )
            vdb.load()
            logger.info("Vector database initialized")

            # 2. 初始化模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lora_checkpoint_path = "./output/checkpoint-73102"

            # 加载配置和基础模型
            config = PeftConfig.from_pretrained(lora_checkpoint_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device
            )

            # 加载LoRA适配器
            global_model = PeftModel.from_pretrained(
                base_model,
                lora_checkpoint_path,
                device_map=device
            )
            global_model.eval()

            # 初始化tokenizer
            global_tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=True,
                pad_token="<|im_end|>"
            )
            if global_tokenizer.pad_token is None:
                global_tokenizer.pad_token = global_tokenizer.eos_token

            logger.info("Model and tokenizer initialized")
        _initialized = True


# 在应用启动时初始化组件
@app.before_request
def before_request():
    if not _initialized:
        initialize_components()

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask():
    """处理问答请求的API端点"""
    global conversation_history  # 声明使用全局历史记录

    try:
        # 解析请求数据
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400

        current_query = data['query']
        logger.info(f"Received query: {current_query}")

        # 数据库检索（保持原有逻辑）
        references = vdb.search(query=current_query)
        contents = [result["text"] for result in references if result['score'] >= 0.5]

        # 构建基础prompt
        system_prompt = prompt_sys(contents)
        base_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # 动态构建历史对话（新增核心逻辑）
        history_prompt = ""
        current_tokens = global_tokenizer(
            base_prompt + f"<|im_start|>user\n{current_query}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.size(1)

        # 逆序添加历史对话直到达到token限制
        keep_history = []
        for i in reversed(range(len(conversation_history))):
            user_msg, assistant_msg = conversation_history[i]
            test_prompt = (
                    f"<|im_start|>history_user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>history_assistant\n{assistant_msg}<|im_end|>\n"
                    + history_prompt
            )
            # 修复token计数方式
            test_tokens = len(global_tokenizer(test_prompt, add_special_tokens=False).input_ids)  # 关键修改点

            if (current_tokens + test_tokens) <= MAX_TOKENS * 0.7:  # 保留30%余量给生成
                history_prompt = test_prompt
                keep_history.insert(0, (user_msg, assistant_msg))
                current_tokens += test_tokens
            else:
                break

            if len(keep_history) >= MAX_HISTORY_LENGTH:
                break

        # 组合最终prompt
        formatted_prompt = (
                base_prompt +
                history_prompt +
                f"<|im_start|>user\n{current_query}<|im_end|>\n"
                "<|im_start|>assistant\n"
        )
        print("formatted_prompt:-------------",formatted_prompt)
        # 生成响应（保持原有逻辑）
        inputs = global_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
            add_special_tokens=False
        ).to(global_model.device)

        with torch.no_grad():
            outputs = global_model.generate(
            ** inputs,
            do_sample = True,
            max_new_tokens = 512,
            temperature = 0.3,
            top_p = 0.85,
            repetition_penalty = 1.2
            )

            # 处理响应（保持原有逻辑）
            full_response = global_tokenizer.decode(
                outputs[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            response_start = full_response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            cleaned_response = full_response[response_start:].split("<|im_end|>")[0].strip()
            cleaned_response = cleaned_response.replace("<|endoftext|>", "")

            # 更新对话历史（新增）
            conversation_history.append((current_query, cleaned_response))

            # 保持历史记录不超过最大限制
            if len(conversation_history) > MAX_HISTORY_LENGTH * 2:  # 两倍缓冲
                conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]

            return jsonify({
                "status": "success",
                "response": cleaned_response,
                "references": '' if not contents or len(contents[0]) < 100 else contents[:3]
            })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    initialize_components()  # 确保直接运行时初始化
    app.run(host='0.0.0.0', port=5000, debug=False)
