from langchain_chroma import Chroma
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
# from transformers import BertTokenizer, BertModel
import torch
from openai import OpenAI

import os
from pathlib import Path
from embed import embed


class ErnieEmbeddingFunction(EmbeddingFunction): 
    def __init__(self, options):
        super().__init__()
        self.options = options
    def embed_documents(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = embed(text, self.options)
            try:
                embedding = response 
                embeddings.append(embedding)
            except (IndexError, TypeError, KeyError) as e:
                print(f"Error processing text: {text}, Error: {e}")
        return embeddings
    def embed_query(self, input) -> Embeddings:
        response = embed(input, self.options)
        try:
            embedding = response 
        except (IndexError, TypeError, KeyError) as e:
            print(f"Error processing text: {input}, Error: {e}")
        return embedding


# 查看向量数据库元素
# all_data = vectordb_chinese.get()
# all_vectors = all_data['embeddings']
# all_metadatas = all_data['metadatas']
# all_document=all_data['documents']
# all_ids = all_data['ids']



# 检索后没有进行排序
def read_from_db(query, k, options):
    persist_directory = options['persist_directory']
    vectordb_chinese = Chroma(
        persist_directory = persist_directory,
        embedding_function = ErnieEmbeddingFunction(options=options)
    )
    retriever = vectordb_chinese.similarity_search(query, k)
    return retriever