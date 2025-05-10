from typing import List, Tuple, Optional
import torch
from sentence_transformers import CrossEncoder


class TextReranker:
    """ 文本重排序器，使用交叉编码器对文本进行相关性打分并排序 """

    def __init__(
            self,
            # 选择 cross-encoder
            model_name: str = "cross-encoder/stsb-distilroberta-base",
            device: Optional[str] = None,
            max_length: int = 512
    ):
        """
        初始化重排序器

        :param model_name: 使用的预训练模型名称，默认为交叉编码模型
        :param device: 指定计算设备（如'cuda', 'cpu'），默认为自动检测
        :param max_length: 文本最大长度，超过将被截断
        """
        # 自动检测GPU设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化交叉编码模型
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
            trust_remote_code=True
        )

        # 记录配置参数
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

    def rerank(
            self,
            query: str,
            texts: List[str],
            top_k: int,
            return_scores: bool = False
    ) -> List[Tuple[str, float]] or List[str]:
        """
        对文本进行重新排序

        :param query: 查询文本
        :param texts: 待排序的文本列表
        :param top_k: 返回前K个结果
        :param return_scores: 是否返回分数
        :return: 排序后的（文本，分数）列表或纯文本列表
        """
        if not texts:
            return []

        # 构造查询-文本对
        query_text_pairs = [(query, text) for text in texts]

        # 获取模型预测分数
        scores = self.model.predict(query_text_pairs)

        # 组合文本与分数
        scored_texts = list(zip(texts, scores))

        # 按分数降序排序
        sorted_results = sorted(scored_texts, key=lambda x: x[1], reverse=True)

        # 取前K个结果
        top_results = sorted_results[:top_k]

        return top_results if return_scores else [text for text, _ in top_results]