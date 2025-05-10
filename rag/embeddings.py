from transformers import AutoModel, AutoTokenizer
from typing import List
import numpy as np
import torch
import os


class Embeddings:
    def __init__(self):
        # 强制禁用Flash Attention相关优化
        os.environ["TORCH_MHLO_ATTENTION"] = "0"  # 新增环境变量设置
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_path = "../models/jina-embeddings-v3"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 强制使用原始Attention实现
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="eager"  # 确保使用非Flash实现
            ).to(self.device)

            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _embed(self, texts: List[str], task_type: str) -> np.ndarray:
        with torch.no_grad():
            # 将task_type写入模型配置
            self.model.config.task_type = task_type  # 新增配置

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192
            ).to(self.device)

            # 不再传递task_type参数
            outputs = self.model(**inputs)  # 移除了task_type参数

            embeddings = outputs.last_hidden_state.float().mean(dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def embed_knowdata(self, content: str) -> dict:
        vectors = self._embed([content], "retrieval_document")
        return {
            "data": [{
                "embedding": vectors[0].tolist(),
                "index": 0
            }]
        }

    def embed_query(self, query: str) -> dict:
        vectors = self._embed([query], "retrieval_query")
        return {
            "data": [{
                "embedding": vectors[0].tolist(),
                "index": 0
            }]
        }
