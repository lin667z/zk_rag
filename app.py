from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag.format_prompt import prompt_sys
from rag.db_store import VectorDatabase
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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
MAX_TOKENS = 8192      # 模型最大token限制
# 在全局变量部分添加
REQUEST_COUNTER = 0
CLEAN_INTERVAL = 3  # 每3次请求清理一次


def log_memory_usage(logger):
    stats = torch.cuda.memory_stats()
    logger.info(f"""
    [Memory] Allocated: {stats['allocated_bytes.all.current']/1024**3:.2f}GB
    Reserved: {stats['reserved_bytes.all.current']/1024**3:.2f}GB
    Active: {stats['active_bytes.all.current']/1024**3:.2f}GB
    Inactive: {stats['inactive_split_bytes.all.current']/1024**3:.2f}GB
    """)


def log_memory_usage(logger):
    stats = torch.cuda.memory_stats()
    logger.info(f"""
    [Memory] Allocated: {stats['allocated_bytes.all.current']/1024**3:.2f}GB
    Reserved: {stats['reserved_bytes.all.current']/1024**3:.2f}GB
    Active: {stats['active_bytes.all.current']/1024**3:.2f}GB
    Inactive: {stats['inactive_split_bytes.all.current']/1024**3:.2f}GB
    """)



def build_history_prompt(conversation_history, tokenizer, base_prompt, user_query, max_context_tokens=5120):
    """动态调整历史对话长度"""
    base_tokens = len(tokenizer.encode(base_prompt + user_query, add_special_tokens=False))
    available_tokens = max_context_tokens - base_tokens - 1024  # 保留生成空间

    history_entries = []
    total_tokens = 0

    # 逆序处理历史记录
    for user_msg, assistant_msg in reversed(conversation_history):
        entry = f"user\n{user_msg}\nassistant\n{assistant_msg}\n"
        entry_tokens = len(tokenizer.encode(entry, add_special_tokens=False))

        if total_tokens + entry_tokens > available_tokens:
            break

        history_entries.append(entry)
        total_tokens += entry_tokens

    return "<|History_start|>"+"".join(reversed(history_entries))+"<|History_end|>"  # 恢复正序




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
    global REQUEST_COUNTER
    CLEAN_INTERVAL = 2
    REQUEST_COUNTER += 1
    if REQUEST_COUNTER % CLEAN_INTERVAL == 0:
        torch.cuda.empty_cache()
        logger.info("Performed periodic memory cleanup")
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

        history_prompt = build_history_prompt(conversation_history, global_tokenizer, base_prompt, current_query)

        # 组合最终prompt
        formatted_prompt = (
                base_prompt +
                history_prompt +
                f"<|im_start|>user\n{current_query}<|im_end|>\n"
                "<|im_start|>assistant\n"
        )
        print("formatted_prompt:-------------\n",formatted_prompt)
        # 生成响应（保持原有逻辑）
        inputs = global_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
            add_special_tokens=False
        ).to(global_model.device)

        # 在生成响应后添加内存监控
        mem_info = torch.cuda.memory_stats()
        logger.info(f"Memory allocated: {mem_info['allocated_bytes.all.current'] / 1024 ** 3:.2f}GB")

        # 修改后的生成部分
        with torch.no_grad():
            outputs = global_model.generate(
            **inputs,
            do_sample = True,
            max_new_tokens = 1024,  # 减少生成长度
            temperature = 0.3,
            top_p = 0.85,
            repetition_penalty = 1.2,
            use_cache = True,
            num_return_sequences = 1  # 确保只返回一个序列
            )

            # 添加张量清理逻辑
            del inputs
            torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            log_memory_usage(logger)
            print("error:\n",str(e))
            return jsonify({
                "status": "success",
                "response": "问题过长，服务器硬件设备性能不足！（温馨提示：可稍后再试。）",
                "references": ''
            })


if __name__ == '__main__':
    initialize_components()  # 确保直接运行时初始化
    app.run(host='0.0.0.0', port=5000, debug=False)
