import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import huggingface_hub

huggingface_hub.login("your_token")
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from typing import List
import time
import random


def monitor_gpu_usage():
    """监测 GPU 显存使用情况"""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # 使用 GPU 0
        info = nvmlDeviceGetMemoryInfo(handle)
        total = info.total / 1024 ** 2  # 总显存（MB）
        used = info.used / 1024 ** 2  # 已用显存（MB）
        free = info.free / 1024 ** 2  # 空闲显存（MB）
        return f"显存总量: {total:.2f} MB, 已用: {used:.2f} MB, 空闲: {free:.2f} MB"
    except Exception as e:
        return f"显存监测失败: {e}"


class DeepseekEmbeddings(Embeddings):
    """自定义 Deepseek 嵌入模型"""

    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        embeddings = []
        for i, text in enumerate(texts):
            # 每处理10个文档打印进度
            if i % 10 == 0:
                print(f"Processing document {i}/{len(texts)}")

            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embedding = self.embed_query(text)
                    embeddings.append(embedding)
                    # 随机延时1-3秒
                    time.sleep(random.uniform(1, 3))
                    break
                except Exception as e:
                    if attempt == max_retries - 1:  # 最后一次尝试
                        raise e
                    print(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                    # 遇到错误时延时更长（5-10秒）
                    time.sleep(random.uniform(5, 10))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "input": text,
            "encoding_format": "float"
        }

        response = requests.post(
            f"{self.api_base}/v1/embeddings",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"Error getting embedding: {response.text}")


def get_pdf_files(pdf_dir):
    """获取目录下的所有PDF文件"""
    return {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}


def print_knowledge_base_info(current_files, new_files=None):
    """打印知识库信息"""
    print("\n" + "=" * 50)
    print("知识库包含以下文件：")
    for file in current_files:
        status = "[新文件]" if new_files and file in new_files else "[已加载]"
        print(f"  {status} {file}")
    print("=" * 50 + "\n")


def load_pdfs(pdf_dir):
    # 确保目录存在
    if not os.path.exists(pdf_dir):
        print(f"创建目录: {pdf_dir}")
        os.makedirs(pdf_dir)

    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cuda',
                      'trust_remote_code': True
                      },
        cache_folder="./embeddings_cache"
    )
    # embeddings = SentenceTransformer(
    #     model_name_or_path="shibing624/text2vec-base-chinese",
    #     backend="onnx",
    #     model_kwargs={"file_name": "model_O4.onnx"},
    #     cache_folder="./embeddings_cache"
    # )

    # 使用 Chroma 存储
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # 检查是否有PDF文件
    current_files = get_pdf_files(pdf_dir)
    if not current_files:
        print(f"\n警告: {pdf_dir} 目录下没有找到PDF文件")
        print(f"请将PDF文件放入 {os.path.abspath(pdf_dir)} 目录")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    # 保存已处理文件列表的路径
    processed_files_path = os.path.join(persist_directory, "processed_files.txt")

    processed_files = set()

    # 读取已处理的文件列表
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r', encoding='utf-8') as f:
            processed_files = set(f.read().splitlines())

    # 检查是否有新文件
    new_files = current_files - processed_files

    # 如果有向量库且没有新文件，直接返回
    if os.path.exists(persist_directory) and not new_files:
        print("Loading existing vector store...")
        print_knowledge_base_info(current_files)
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    # 处理新文件或创建新的向量库
    documents = []

    if os.path.exists(persist_directory) and new_files:
        print("发现新的PDF文件，正在更新知识库...")
        print_knowledge_base_info(current_files, new_files)
        # 加载现有向量库
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # 只处理新文件
        for file in new_files:
            print(f"Loading new PDF: {file}")
            loader = PyMuPDFLoader(os.path.join(pdf_dir, file))
            documents.extend(loader.load())
    else:
        print("Creating new vector store...")
        print_knowledge_base_info(current_files)
        # 处理所有文件
        for file in current_files:
            print(f"Loading PDF: {file}")
            loader = PyMuPDFLoader(os.path.join(pdf_dir, file))
            documents.extend(loader.load())

    if documents:  # 如果有新文档需要处理
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")

        batch_size = 1  # 每次处理 10 个 chunk
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                print(f"正在处理 batch {i + 1}/{len(texts)}")
                print(monitor_gpu_usage())  # 显存状态
                if os.path.exists(persist_directory):
                    # 将新文档添加到现有向量库
                    vectorstore.add_documents(batch)  # 添加到向量库
                    torch.cuda.empty_cache()  # 释放显存
                else:
                    # 创建新的向量库
                    vectorstore = Chroma.from_documents(
                        documents=texts,
                        embedding=embeddings,
                        persist_directory=persist_directory
                    )
            except Exception as e:
                print(f"处理 batch {i} 时出错: {e}")
                break

        # if os.path.exists(persist_directory):
        #     # 将新文档添加到现有向量库
        #     vectorstore.add_documents(texts)
        # else:
        #     # 创建新的向量库
        #     vectorstore = Chroma.from_documents(
        #         documents=texts,
        #         embedding=embeddings,
        #         persist_directory=persist_directory
        #     )

        # 更新已处理文件列表
        processed_files.update(current_files)
        with open(processed_files_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_files))

    return vectorstore
