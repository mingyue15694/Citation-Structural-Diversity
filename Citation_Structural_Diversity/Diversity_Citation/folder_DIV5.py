# 组合语义增强引文网络模型的结构多样性
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

year_data = {}
# 从 diversity_data202.json 中加载发表年份
with open('date_clean_later8.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        year_data[data["paper_id"]] = int(data["year"])


def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    dot_product = torch.dot(embedding1, embedding2)
    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)
    return (dot_product / (norm1 * norm2)).item()


# 从文件夹中读取嵌入向量
def load_embedding(embedding_files_folder, node_id):
    find_file = node_id // 100000 + 1
    find_line = node_id % 100000
    if find_line == 0:
        find_file = 100000
        find_file -= 1
    if find_file > 36 or node_id > 3509437:
        return 0, 0
    embedding_file_path = os.path.join(r'output_embeddings4',
                                       f'output_embeddings{find_file}.json')

    with open(embedding_file_path, 'r', encoding='utf-8') as embedding_file:
        for _ in range(find_line - 1):
            next(embedding_file)
        data = json.loads(next(embedding_file))

    paper_id = data["paper_id"]
    embedding = np.array(data["embedding"])
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
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
    G = nx.DiGraph()
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    paper_id = entry['paper_id']
                    references = entry.get('references', [])
                    if paper_id not in G:
                        G.add_node(paper_id)
                    for ref_id in references:
                        if ref_id not in G:
                            G.add_node(ref_id)
                        G.add_edge(paper_id, ref_id)
    return G


def build_temp_graph(G, embedding_files_folder, output_file):
    first_entry = True  # 添加这行以定义并初始化变量first_entry
    # 在循环迭代图 G 的节点时打开 JSON 文件进行追加写入
    with open(output_file, 'a', encoding='utf-8') as json_file:
        for node_id in G:
            if node_id not in paper_ids:
                continue
            print(node_id)
            diversity_data = {}  # 创建一个空的数据字典，用于存储当前节点的数据
            embeddings = {}
            d1 = 1
            d2 = 6
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
                    node_temp_id, embedding_tensor = load_embedding(embedding_files_folder, node_temp_id)
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
                        common_successors = [p for p in common_successors if p != node_id]
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

            if Similarity1:

                # 计算相似度数据的平均值
                avg_similarity1 = np.mean(Similarity1)

                # 将这些统计值作为相似度阈值
                similarity_thresholds1 = [avg_similarity1]

            else:
                # 如果没有相似度数据，设置一个默认的阈值
                similarity_thresholds1 = [0.0]

            all_similarity_thresholds = similarity_thresholds1

            # 添加相似性边
            for similarity_threshold in all_similarity_thresholds:
                G_copy = G_temp.copy()
                for paper1 in G_copy.nodes():
                    for paper2 in G_copy.nodes():
                        if paper1 != paper2:  # 确保不对自身添加边
                            embedding1 = embeddings.get(paper1)
                            embedding2 = embeddings.get(paper2)
                            if embedding1 is not None and embedding2 is not None:
                                similarity = cosine_similarity(embedding1, embedding2)
                                if similarity > similarity_threshold:
                                    # 如果相似度超过阈值，则添加紫色边
                                    if not G_copy.has_edge(paper1, paper2):
                                        G_copy.add_edge(paper1, paper2, color='purple')
                                    # elif not G_copy.has_edge(paper2, paper1):
                                    #     G_copy.add_edge(paper2, paper1, color='purple')

            # 计算节点的结构多样性
            node_diversity1 = calculate_structural_diversity1(G_temp, node_id)

            node_data = {
                "paper_id": node_id,
                f"node_diversity{d1}": node_diversity1,

            }

            diversity_data[node_id] = node_data

            # 将每个节点的数据写入 JSON 文件
            if not first_entry:
                json_file.write(',\n')  # 如果不是第一个条目，添加逗号分隔符
            json.dump({node_id: diversity_data}, json_file, indent=4)
            first_entry = False
        json_file.write(']\n')


# 从文件夹中读取嵌入向量
embedding_files_folder = 'output_embeddings4'

# 读取JSON文件并构建引文网络结构图
data_folder = 'output_date_files3_largest_component'
G = build_citation_network(data_folder)
output_folder = 'year_venue_50_div5'
os.makedirs(output_folder, exist_ok=True)  # 确保输出目录存在

data_text_folder = 'year_venue_data'
for filename in os.listdir(data_text_folder):
    if filename.endswith(".json"):
        paper_ids = []
        file_path = os.path.join(data_text_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                paper_ids.append(data["paper_id"])

        # 生成输出文件路径
        output_file = os.path.join(output_folder, filename)
        build_temp_graph(G, embedding_files_folder, output_file)
