from typing import List, Union, Tuple, Dict
from tqdm import tqdm
import os
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    """支持批量文件处理的文档加载器"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        初始化文档加载器
        :param chunk_size: 文本块大小（字符数）
        :param chunk_overlap: 块间重叠大小（字符数）
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )

        # 支持的文件类型映射
        self.supported_extensions = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".html": UnstructuredHTMLLoader,
            ".md": UnstructuredMarkdownLoader,
        }

    def load_document(self, file_path: Union[str, Path], max_size_mb: int = 50) -> List[Document]:
        """
        加载并处理单个文档文件
        :param file_path: 文档路径
        :param max_size_mb: 文件最大允许大小（MB）
        :return: 分块后的Document对象列表
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = path.suffix.lower()
        if file_extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {file_extension}\n"
                             f"支持格式: {', '.join(self.supported_extensions.keys())}")

        file_size_mb = path.stat().st_size / (1024 * 1024)  # MB
        if file_size_mb > max_size_mb:
            print(f"警告: 大文件 {path.name} ({file_size_mb:.1f}MB)")

        loader_class = self.supported_extensions[file_extension]
        loader = loader_class(str(path))
        docs = loader.load()

        full_text = "\n\n".join([doc.page_content for doc in docs])
        metadata = {"source": str(path).split("\\")[-1]}
        full_doc = [Document(page_content=full_text, metadata=metadata)]

        split_docs = self.text_splitter.split_documents(full_doc)
        return split_docs

    def load_directory(
            self,
            directory: Union[str, Path],
            recursive: bool = False,
            ignore_errors: bool = True
    ) -> Tuple[List[Document], Dict[str, str]]:
        """
        批量加载目录中的文档
        :param directory: 目录路径
        :param recursive: 是否递归处理子目录
        :param ignore_errors: 是否忽略错误文件
        :return: 成功加载的文档列表, 错误文件字典
        """
        path = Path(directory)
        if not path.is_dir():
            raise ValueError(f"路径不是目录: {directory}")

        all_docs = []
        error_files = {}

        file_paths = []
        for root, _, files in os.walk(path):
            if not recursive and root != str(path):
                continue
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_extensions:
                    file_paths.append(file_path)

        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                docs = self.load_document(file_path)
                all_docs.extend(docs)
            except Exception as e:
                error_files[str(file_path)] = str(e)
                if not ignore_errors:
                    raise
                else:
                    print(f"Error processing {file_path}: {e}")

        return all_docs, error_files


if __name__ == "__main__":
    loader = DocumentLoader(chunk_size=800, chunk_overlap=50)

    try:
        docs, errors = loader.load_directory("../data/data_base", recursive=True)

        print(f"\n成功处理 {len(docs)} 个文本块")
        print(f"发现 {len(errors)} 个错误文件")

        if errors:
            print("\n错误文件列表:")
            for file, error in errors.items():
                print(f"- {file}: {error}")

        # if docs:
        #     print("\n前3个文本块示例:")
        #     for i, doc in enumerate(docs[:3], 1):
        #         print(f"\n文本块 #{i}:")
        #         print(doc.page_content[:200] + "...")
        #         print(f"元数据: {doc.metadata}")
        #     print(docs)
    except Exception as e:
        print(f"处理失败: {str(e)}")


