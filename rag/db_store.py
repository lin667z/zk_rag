import numpy as np
import faiss
import pickle
import traceback
from pathlib import Path
from typing import List, Dict, Union, Optional

from rag.embeddings import Embeddings
from rag.docu_load import DocumentLoader


class VectorDatabase:
    def __init__(self, index_path: str = "./cache/faiss_index.bin",
                 metadata_path: str = "./cache/metadata.pkl"):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.dim = 1024  # 必须与Jina模型维度匹配

        # 初始化本地嵌入模型
        self.emb = Embeddings()

    def build_from_directory(self, directory: Union[str, Path]):
        """构建向量数据库（分批次处理）"""
        d = DocumentLoader(chunk_size=512, chunk_overlap=50)
        docs, errors = d.load_directory(directory)
        print(f"\n成功处理 {len(docs)} 个文本块")
        print(f"发现 {len(errors)} 个错误文件")

        self.metadata = []
        self.index = faiss.IndexFlatIP(self.dim)
        batch_size = 50
        total_processed = 0

        for batch_idx in range(0, len(docs), batch_size):
            batch = docs[batch_idx: batch_idx + batch_size]
            batch_vectors = []
            batch_metadata = []
            processed = 0

            for doc in batch:
                try:
                    content = doc.page_content
                    metadata = doc.metadata

                    # 调用本地模型
                    embedding_response = self.emb.embed_knowdata(content)

                    if not embedding_response or 'data' not in embedding_response:
                        print(f"无效的embedding响应: {embedding_response}")
                        continue

                    embedding_data = embedding_response['data'][0]['embedding']
                    if len(embedding_data) != self.dim:
                        print(f"维度异常: 期望 {self.dim} 实际 {len(embedding_data)}")
                        continue

                    embedding_vector = np.array(embedding_data, dtype=np.float32)
                    batch_vectors.append(embedding_vector)
                    batch_metadata.append({
                        "text": content,
                        "source": metadata.get("source"),
                    })
                    processed += 1
                    total_processed += 1

                except Exception as e:
                    print(f"\n处理文档出错：{str(e)}")
                    traceback.print_exc()

            # 添加当前批次数据
            if batch_vectors:
                vectors_np = np.vstack(batch_vectors)
                faiss.normalize_L2(vectors_np)
                self.index.add(vectors_np)
                self.metadata.extend(batch_metadata)
                print(f"已处理批次 [{batch_idx // batch_size + 1}] 成功插入 {processed} 个向量")

        # 保存数据
        if self.index.ntotal > 0:
            self._save()
            print(f"数据库构建完成，共索引 {total_processed} 个文本块")
        else:
            print("错误：未能生成有效索引，可能原因：")
            print("1. 所有文档处理失败")
            print("2. 嵌入维度不匹配（当前设置dim=1024）")
            print("3. 模型路径配置错误")
            raise RuntimeError("索引构建失败")

    def _save(self):
        """保存索引和元数据"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """加载已有数据库"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int = 1) -> List[Dict]:
        """语义检索"""
        if self.index is None:
            raise ValueError("索引未初始化，请先加载或构建数据库")

        try:
            # 调用本地模型
            query_response = self.emb.embed_query(query)

            query_data = query_response['data'][0]['embedding']
            query_vec = np.array(query_data, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vec)

            distances, indices = self.index.search(query_vec, top_k)

            return [{
                "text": self.metadata[idx]["text"],
                "source": self.metadata[idx]["source"],
                "score": float(distances[0][i])
            } for i, idx in enumerate(indices[0]) if idx != -1]

        except Exception as e:
            print(f"查询过程中发生错误: {str(e)}")
            return []


def main():
    # 初始化向量数据库
    vdb = VectorDatabase(
        index_path="cache/faiss_index.bin",
        metadata_path="cache/metadata.pkl"
    )

    # 数据目录路径（使用原始字符串）
    data_dir = Path(r"../dataset/data_base")

    # 构建索引（首次运行）
    try:
        if not vdb.index_path.exists():
            print("开始构建向量数据库...")
            vdb.build_from_directory(data_dir)
        else:
            print("检测到已有索引文件")
    except Exception as e:
        print(f"构建过程中发生错误: {str(e)}")
        return

    # 加载已有索引
    try:
        print("\n加载数据库...")
        vdb.load()
        print(f"成功加载数据库，现有索引数量: {vdb.index.ntotal}")
    except Exception as e:
        print(f"加载失败: {str(e)}")
        return

    # 执行查询
    while True:
        query = input("\n请输入查询内容（输入q退出）: ")
        if query.lower() == 'q':
            break

        try:
            results = vdb.search(query, top_k=3)
            if not results:
                print("未找到相关结果")
                continue

            print(f"\n找到 {len(results)} 个相关结果：")
            for i, res in enumerate(results, 1):
                print(f"{i}. [相关性：{res['score']:.3f}] 来源：{res['source']}")
                print(f"   {res['text'][:150]}...\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"查询失败: {str(e)}")


if __name__ == "__main__":
    main()