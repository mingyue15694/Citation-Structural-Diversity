# 语义组合
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def plot_citation_network(G):
    if len(G) == 0:
        print("G_temp is empty, cannot plot.")
        return
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    # 对节点标签进行处理，只显示后三位
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    for u, v, d in G.edges(data=True):
        style = d.get('style', '-')  # 指定默认样式为实线
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.7, width=1, edge_color=d.get('color'), style=style)

    plt.title("Citation Network")
    plt.axis('off')
    plt.show()


year_data = {}
# 从 diversity_data202.json 中加载发表年份
with open('F:/mingyue/specter/pythonProject1/date_clean_later8.json', 'r', encoding='utf-8') as f:
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
    embedding_file_path = os.path.join(r'F:\mingyue\specter\pythonProject1\output_embeddings4',
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

            # 添加节点 node_id 的入度节点
            in_neighbors = [in_neighbor for in_neighbor in G.predecessors(node_id)
                            if abs(year_data.get(in_neighbor, 0) - year_data.get(node_id, 0)) <= 3]
            if len(in_neighbors) == 0:
                continue
            G_temp.add_nodes_from(in_neighbors)  # 将入度节点添加到子图

            # 提取图 G_temp 中的文献的 embedding
            for node_temp_id in G_temp:
                if node_temp_id not in embeddings:  # 如果节点的嵌入向量尚未计算过
                    node_temp_id, embedding_tensor = load_embedding(embedding_files_folder, node_temp_id)
                    if node_temp_id == 0:
                        continue
                    embeddings[node_temp_id] = embedding_tensor

            # 添加节点 node_id 的入度边
            for in_neighbor in in_neighbors:
                G_temp.add_edge(in_neighbor, node_id)
                embedding_u = embeddings.get(in_neighbor)
                embedding_v = embeddings.get(node_id)
                if embedding_u is not None and embedding_v is not None:
                    similarity = cosine_similarity(embedding_u, embedding_v)
                    Similarity1.append(similarity)

            # 添加入度节点之间的边
            for i, in_neighbor1 in enumerate(in_neighbors):
                for in_neighbor2 in in_neighbors[i + 1:]:
                    if G.has_edge(in_neighbor1, in_neighbor2):
                        G_temp.add_edge(in_neighbor1, in_neighbor2)
                        embedding_u = embeddings.get(in_neighbor1)
                        embedding_v = embeddings.get(in_neighbor2)
                        if embedding_u is not None and embedding_v is not None:
                            similarity = cosine_similarity(embedding_u, embedding_v)
                            Similarity2.append(similarity)
                    elif G.has_edge(in_neighbor2, in_neighbor1):
                        G_temp.add_edge(in_neighbor2, in_neighbor1)
                        embedding_u = embeddings.get(in_neighbor2)
                        embedding_v = embeddings.get(in_neighbor1)
                        if embedding_u is not None and embedding_v is not None:
                            similarity = cosine_similarity(embedding_u, embedding_v)
                            Similarity2.append(similarity)

            # 添加耦合边（包括出度节点的出度节点）
            for out_neighbor1 in in_neighbors:
                successors1 = set(G.successors(out_neighbor1))
                for out_neighbor2 in in_neighbors:
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
            for i, out_neighbor1 in enumerate(in_neighbors):
                predecessors1 = set(G.predecessors(out_neighbor1))
                year1 = year_data.get(out_neighbor1, 0)
                for j, out_neighbor2 in enumerate(in_neighbors):
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


            # 将当前节点的数据写入 JSON 文件
            if first_entry:
                # json_file.write('[')
                first_entry = False
            else:
                json_file.write(',\n')
            json.dump({node_id: diversity_data}, json_file, indent=4)
        json_file.write(']\n')


# 从文件夹中读取嵌入向量
embedding_files_folder = 'F:/mingyue/specter/pythonProject1/output_embeddings4'

# 读取JSON文件并构建引文网络结构图
data_folder = 'F:/mingyue/specter/pythonProject1/output_date_files3_largest_component'
G = build_citation_network(data_folder)
output_folder = 'F:/mingyue/specter/pythonProject1/seven_div/year_venue_50_div2_2'
os.makedirs(output_folder, exist_ok=True)  # 确保输出目录存在

data_text_folder = 'F:/mingyue/specter/pythonProject1/year_venue_50_1'
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
        build_temp_graph(G, embedding_files_folder,output_file)

