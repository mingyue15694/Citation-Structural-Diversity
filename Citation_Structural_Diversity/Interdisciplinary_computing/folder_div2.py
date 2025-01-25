# 语义组合
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools
import os
from itertools import product

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    dot_product = torch.dot(embedding1, embedding2)
    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)
    return (dot_product / (norm1 * norm2)).item()


# 从文件夹中读取嵌入向量
def load_embedding(embedding_file_path, node_id):
    paper_id = None  # 初始化 paper_id
    embedding_tensor = None  # 初始化 embedding_tensor

    # 检查文件路径是否存在
    if os.path.exists(embedding_file_path):
        with open(embedding_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                paper_id = int(data["paper_id"])
                if node_id == paper_id:
                    embedding = np.array(data["embedding"])
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32, requires_grad=True)
                    break  # 找到后可以提前退

    # # 检查是否找到了匹配的 embedding_tensor
    # if embedding_tensor is None:
    #     raise ValueError(f"Embedding for node_id {node_id} not found.")

    return paper_id, embedding_tensor


def calculate_structural_diversity1(graph, node_u):
    """
    计算节点的结构多样性，方法1：
    从指定节点的邻居节点开始，构建子图，计算其中连通分量的数量作为结构多样性指标。
    """
    neighbors = list(graph.neighbors(node_u))  # 获取指定节点的邻居节点列表
    if not neighbors:  # 如果指定节点没有邻居节点
        return 0
    subgraph_nodes = set(neighbors)  # 初始化子图节点集合为邻居节点集合

    subgraph = graph.subgraph(subgraph_nodes).copy()  # 构建子图
    number_of_components = nx.number_connected_components(subgraph)  # 计算子图的连通分量数量作为结构多样性指标
    return number_of_components

# 构建引文网络
def build_citation_network(data_folder):
    G = nx.DiGraph()  # 创建有向图
    for filename in os.listdir(data_folder):  # 遍历文件夹中的文件
        if filename.endswith(".json"):  # 只处理 .json 文件
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    paper_id = entry['PMID']  # 获取文章 ID
                    # try:
                    #     # paper_id = int(paper_id)  # 尝试转换为整数
                    # except ValueError:
                    #     print(f"Invalid paper ID: {paper_id}, skipping.")
                    #     continue  # 跳过无效的 ID

                    references = entry.get('References', [])  # 获取引用列表
                    if paper_id not in G:  # 如果节点不存在，则添加
                        G.add_node(paper_id)

                    for ref_id in references:
                        try:
                            # ref_id = int(ref_id)  # 尝试将引用 ID 转为整数
                            if ref_id not in G:
                                G.add_node(ref_id)
                            G.add_edge(paper_id, ref_id)  # 添加边
                        except ValueError:
                            print(paper_id)
                            print(f"Invalid reference ID: {ref_id}, skipping.")
                            continue  # 跳过无效的引用 ID
    return G


def build_temp_graph(G, embedding_file_path, output_file):
    first_entry = True
    # 打开 JSON 文件进行写入
    with open(output_file, 'w', encoding='utf-8') as json_file:
        for node_id in G:
            if node_id not in paper_ids:
                continue
            print(node_id)
            diversity_data = {}  # 创建一个空的数据字典，用于存储当前节点的数据
            embeddings = {}
            d1 = 1
            d2 = 16
            # 新建一个空图 G_temp
            G_temp = nx.Graph()  # 将有向图改为无向图
            G_temp.add_node(node_id)
            #  node_id 和其参考文献之间的相似度
            Similarity1 = []
            # 其参考文献和参考文献之间的相似度
            Similarity2 = []

            # 添加节点 node_id 的出度节点
            out_neighbors = list(G.successors(node_id))
            if len(out_neighbors) == 0:
                continue
            G_temp.add_nodes_from(out_neighbors)

            # 提取图 G_temp 中的文献的 embedding
            for node_temp_id in G_temp:
                if node_temp_id not in embeddings:  # 如果节点的嵌入向量尚未计算过
                    node_temp_id, embedding_tensor = load_embedding(embedding_file_path, node_temp_id)
                    if node_temp_id == 0:
                        continue
                    embeddings[node_temp_id] = embedding_tensor

            # 添加节点 node_id 的出度边
            for out_neighbor in out_neighbors:
                G_temp.add_edge(node_id, out_neighbor)
                embedding_u = embeddings.get(node_id)
                embedding_v = embeddings.get(out_neighbor)
                if embedding_u is not None and embedding_v is not None:
                    similarity = cosine_similarity(embedding_u, embedding_v)
                    Similarity1.append(similarity)

            # 添加出度节点之间的边
            for i, out_neighbor1 in enumerate(out_neighbors):
                for out_neighbor2 in out_neighbors[i + 1:]:
                    if G.has_edge(out_neighbor1, out_neighbor2):
                        G_temp.add_edge(out_neighbor1, out_neighbor2)
                        embedding_u = embeddings.get(out_neighbor1)
                        embedding_v = embeddings.get(out_neighbor2)
                        if embedding_u is not None and embedding_v is not None:
                            similarity = cosine_similarity(embedding_u, embedding_v)
                            Similarity2.append(similarity)
                    elif G.has_edge(out_neighbor2, out_neighbor1):
                        G_temp.add_edge(out_neighbor2, out_neighbor1)
                        embedding_u = embeddings.get(out_neighbor2)
                        embedding_v = embeddings.get(out_neighbor1)
                        if embedding_u is not None and embedding_v is not None:
                            similarity = cosine_similarity(embedding_u, embedding_v)
                            Similarity2.append(similarity)

            # 添加耦合边（包括出度节点的出度节点）
            for out_neighbor1 in out_neighbors:
                successors1 = set(G.successors(out_neighbor1))
                for out_neighbor2 in out_neighbors:
                    if out_neighbor1 != out_neighbor2:
                        successors2 = set(G.successors(out_neighbor2))
                        common_successors = successors1.intersection(successors2)
                        if common_successors:
                            if not G_temp.has_edge(out_neighbor1, out_neighbor2):
                                G_temp.add_edge(out_neighbor1, out_neighbor2, color='green', style='dotted')
                                embedding_u = embeddings.get(out_neighbor2)
                                embedding_v = embeddings.get(out_neighbor1)
                                if embedding_u is not None and embedding_v is not None:
                                    similarity = cosine_similarity(embedding_u, embedding_v)
                                    Similarity2.append(similarity)

            # 添加共被引边
            for i, out_neighbor1 in enumerate(out_neighbors):
                predecessors1 = set(G.predecessors(out_neighbor1))
                year1 = year_data.get(out_neighbor1, 0)
                for j, out_neighbor2 in enumerate(out_neighbors):
                    if i != j:
                        predecessors2 = set(G.predecessors(out_neighbor2))
                        year2 = year_data.get(out_neighbor2, 0)

                        common_predecessors = predecessors1.intersection(predecessors2)
                        common_predecessors.discard(node_id)
                        for common_pred in common_predecessors:
                            year3 = year_data.get(common_pred, 0)
                            if abs(year3 - year2) <= 3 and abs(year3 - year1) <= 3:
                                if not G_temp.has_edge(out_neighbor1, out_neighbor2):
                                    G_temp.add_edge(out_neighbor1, out_neighbor2, color='blue', style='dashed')
                                    embedding_u = embeddings.get(out_neighbor2)
                                    embedding_v = embeddings.get(out_neighbor1)
                                    if embedding_u is not None and embedding_v is not None:
                                        similarity = cosine_similarity(embedding_u, embedding_v)
                                        Similarity2.append(similarity)



            # 计算节点的结构多样性
            node_diversity1 = calculate_structural_diversity1(G_temp, node_id)


            node_data = {
                "paper_id": node_id,
                f"node_diversity{d1}": node_diversity1,

            }

            diversity_data[node_id] = node_data

            if first_entry:
                json_file.write('[')
                first_entry = False
            else:
                json_file.write(',\n')
            print(diversity_data)
            json.dump({node_id: diversity_data}, json_file, indent=4)
        json_file.write(']\n')
        print(f"Output file will be saved to: {output_file}")




# 读取JSON文件并构建引文网络结构图
data_folder = 'F:/mingyue/specter/pythonProject1/pubmed_data'
G = build_citation_network(data_folder)
# 找出提取的id，确保id和参考文献都是可计算的，如果不可计算，则按照0处理
extract_ids = []
# 遍历结构多样性值文件夹中的所有文件
for filename in os.listdir(data_folder):
    if filename.endswith(".json"):
        # print(filename)
        file_path = os.path.join(data_folder, filename)

        # 加载文件内容，期望内容为列表
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        # 如果file_data是列表，逐项遍历以提取信息
        if isinstance(file_data, list):
            for entry in file_data:
                paper_id = entry.get("PMID", "")
                extract_ids.append((paper_id))
output_folder = 'F:/mingyue/specter/pythonProject1/pubmed_div2'
os.makedirs(output_folder, exist_ok=True)  # 确保输出目录存在

data_text_folder = 'F:/mingyue/specter/pythonProject1/pubmed_data_4'
for filename in os.listdir(data_text_folder):
    if filename.endswith(".json"):
        paper_ids = []
        year_data = {}
        file_path = os.path.join(data_text_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            # 直接加载整个 JSON 文件
            data_list = json.load(f)
        # 提取所有文章的paper_id和引用ID
        for data in data_list:
            paper_id = data.get("PMID", "")
            if paper_id:
                paper_ids.append(paper_id)
            year = data.get("ArticleDate", "")
            year_data[paper_id] = int(year)  # 提取 paper_id 和年份
            # print(year_data)
        # 生成输出文件路径
        output_file = os.path.join(output_folder, filename)
        embedding_file_path = os.path.join(r'F:\mingyue\specter\pythonProject1\pubmed_embeddings',
                                           f'embedding_{filename}')
        print(embedding_file_path)
        build_temp_graph(G, embedding_file_path,  output_file)
