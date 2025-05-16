import os
import json
import uuid
from pathlib import Path
from langchain_chroma import Chroma
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from langchain_core.documents import Document

# API设置
# base_url = "https://opus.gptuu.com/v1"
# api_key = "sk-8wjtgEDWLcgAgTrLRNxFUB6E7yUr5iu9H4cOum7TQXSAL2Ny"
# embedding_model = 'text-embedding-3-large'
# persist_directory_chinese = "./db/xldatabase/rag"

# 用于计数
counter = 0

def embed(content):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(input=content, model=embedding_model).data[0].embedding
    return response

class ErnieEmbeddingFunction(EmbeddingFunction):
    def embed_documents(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = embed(text)
            global counter
            counter += 1
            print(f'{counter} / {len(input)} 文档正在嵌入...')
            try:
                embedding = response
                embeddings.append(embedding)
            except (IndexError, TypeError, KeyError) as e:
                print(f"处理文本时出错: {text}, 错误: {e}")
        return embeddings

    def embed_query(self, input) -> Embeddings:
        response = embed(input)
        try:
            embedding = response
        except (IndexError, TypeError, KeyError) as e:
            print(f"处理查询时出错: {input}, 错误: {e}")
        return embedding

# 读取指定路径下的所有 JSON 文件，并保留文件夹结构信息
def read_json_files(json_files_path):
    print('Reading JSON files...')
    documents = []

    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(json_files_path, topdown=True):
        for file in files:
            if file.endswith('.json'):
                file_path = Path(root) / file
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        # 读取 JSON 文件内容
                        content = json.load(f)

                        # 提取 JSON 数据中的 Document 列表
                        document_list = content.get('content', {}).get('Document', [])

                        # 遍历 Document 列表，提取每个文档的 metadata 和 page_content
                        for doc in document_list:
                            page_content = doc.get('page_content', '')
                            metadata = doc.get('metadata', '')

                            # 如果 metadata 是字符串，包装为字典
                            if isinstance(metadata, str):
                                metadata = {'content': metadata}

                            # 确保 metadata 是字典类型
                            if not isinstance(metadata, dict):
                                metadata = {}

                            # 创建 Document 对象
                            document = Document(
                                page_content=page_content,
                                metadata=metadata
                            )
                            documents.append(document)

                        # 额外的元数据（例如 time, source, link 等）
                        additional_metadata = {
                            'time': content.get('time', ''),
                            'source': content.get('source', ''),
                            'link': content.get('link', '')
                        }

                        # 如果需要将额外的元数据添加到最后一个文档对象的元数据中
                        if documents:
                            documents[-1].metadata.update(additional_metadata)

                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")

    print('Reading JSON files done!')
    return documents

# 假设你指定了一个 JSON 文件夹路径
json_files_path = 'D:/TuGraph_RAG-main/data/data_json'

# 读取 JSON 文件
json_knowledge = read_json_files(json_files_path)

# 确保文档不为空
if not json_knowledge:
    print("没有读取到有效的文档内容！")
else:
    # 存入向量数据库
    vectordb_chinese = Chroma.from_documents(
        documents=json_knowledge,
        embedding=ErnieEmbeddingFunction(),
        persist_directory = persist_directory_chinese
    )
    print("向量数据库存储完成！")