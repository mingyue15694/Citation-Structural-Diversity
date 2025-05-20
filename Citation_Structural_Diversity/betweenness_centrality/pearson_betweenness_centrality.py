import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os
import networkx as nx
from torch import tensor
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 构建引文网络
def build_citation_network(data_folder):
    G = nx.DiGraph()
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    paper_id = entry['PMID']
                    references = entry.get('References', [])
                    if paper_id not in G:
                        G.add_node(paper_id)
                    for ref_id in references:
                        if ref_id not in G:
                            G.add_node(ref_id)
                        G.add_edge(paper_id, ref_id)
    return G


data_folder = 'F:/mingyue/specter/pythonProject1/pubmed_data'
G = build_citation_network(data_folder)

# 结构多样性值的文件夹
date_folder = 'F:/mingyue/specter/pythonProject1/pubmed_div31'

# 引文量的文件夹
venue_folder = 'F:/mingyue/specter/pythonProject1/pubmed_data'

# 初始化用于整个文件夹的多样性值和引用量
all_diversity_values = []
all_venue_values = []

year_data = {}
# 加载发表年份
for filename in os.listdir(venue_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(venue_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)  # 假设是一个列表类型的 JSON
            for data in data_list:
                pmid = data.get("PMID")
                year = data.get("ArticleDate")
                if pmid and year:
                    year_data[pmid] = int(year)

def local_betweenness(G, root_pid, depth=2, k=50, seed=42):
    # 广度优先搜索 收集深度至多 depth 的节点
    visited = {root_pid}
    current_layer = {root_pid}
    for _ in range(depth):
        next_layer = set()
        for u in current_layer:
            neighbors = set(G.successors(u)) | set(G.predecessors(u))
            for v in neighbors:
                if v not in visited:
                    visited.add(v)
                    next_layer.add(v)
        current_layer = next_layer
        if not current_layer:
            break
    # 构建无向子图
    H = nx.Graph()
    H.add_nodes_from(visited)
    for u in visited:
        for v in (set(G.successors(u)) | set(G.predecessors(u))):
            if v in visited:
                H.add_edge(u, v)
    # 近似中介中心性，采样 k 个节点
    k_sample = min(k, H.number_of_nodes())
    bc = nx.betweenness_centrality(H, k=k_sample, normalized=True, seed=seed)
    return bc.get(root_pid, 0)

# print(year_data)
# # 遍历结构多样性值文件夹中的所有文件
for filename in os.listdir(date_folder):
    if filename.endswith(".json"):
        # 初始化多样性字典
        print(filename)
        diversity_values = {f"diversity_values_{i}": [] for i in range(3, 4)}
        venue_values = []
        paper_ids = []
        file_path = os.path.join(date_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            paper_id = None
            for line in f:
                # 1) 匹配 "paper_id": "16539706" （ID 始终用双引号包裹）
                m_id = re.search(r'"paper_id"\s*:\s*"(\d+)"', line)
                if m_id:
                    paper_id = int(m_id.group(1))
                    if paper_id in paper_ids:
                        continue
                    paper_ids.append(paper_id)
                    continue  # 拿到 ID 后继续下一行匹配多样性

                # 2) 匹配 "node_diversityX": 59  （X 为任意数字，多样性值无引号）
                m_div = re.search(r'"node_diversity(\d+)"\s*:\s*(\d+)', line)
                if m_div and paper_id is not None:
                    idx = int(m_div.group(1))
                    if idx != 3:
                        continue
                    value = int(m_div.group(2))
                    key = f"diversity_values_{idx}"

                    # 初始化列表并追加
                    diversity_values.setdefault(key, []).append({paper_id: value})
        #
        print(diversity_values)
        # # 读取引用量文件
        #
        # print(paper_ids)

        venue_values = []
        for pid in paper_ids:
            if not G.has_node(str(pid)):
                continue
            # 只考虑出度 >= 5 的节点
            if G.out_degree(str(pid)) < 5:
                continue
            bc = local_betweenness(G, str(pid), depth=2, k=50, seed=42)
            venue_values.append({pid: bc})
        print(f"  computed betweenness for {len(venue_values)} nodes")



        for i in range(3, 4):
            # 将当前文件的数据添加到总集合中
            # all_diversity_values.extend(diversity_values[f"diversity_values_{i}"])
            # all_venue_values.extend(venue_values)
            all_diversity = []
            all_citations = []
            grouped_diversity = {'low': [], 'medium': [], 'high': []}
            grouped_median_citations = {'low': [], 'medium': [], 'high': []}
            diversity_number = {'low': 0, 'medium': 0, 'high': 0}
            key = f"diversity_values_{i}"
            diversity_list = diversity_values[key]
            diversity_groups = {}
            citation_groups = {}
            citation_dict = {list(entry.keys())[0]: list(entry.values())[0] for entry in venue_values}

            # 将多样性值与引用量对应起来
            for diversity_value in diversity_list:
                paper_id = list(diversity_value.keys())[0]
                diversity1 = list(diversity_value.values())[0]
                citation_count = citation_dict.get(paper_id, None)
                if citation_count is not None:
                    all_diversity.append(diversity1)
                    all_citations.append(citation_count)
                    if diversity1 in diversity_groups:
                        diversity_groups[diversity1].append(paper_id)
                        citation_groups[diversity1].append(citation_count)
                    else:
                        diversity_groups[diversity1] = [paper_id]
                        citation_groups[diversity1] = [citation_count]

            # 统计每个多样性组的文献数量
            for diversity_group in diversity_groups:
                if 1 <= diversity_group <= 3:
                    diversity_number['low'] += len(diversity_groups[diversity_group])
                elif 4 <= diversity_group <= 6:
                    diversity_number['medium'] += len(diversity_groups[diversity_group])
                elif diversity_group >= 7:
                    diversity_number['high'] += len(diversity_groups[diversity_group])
            print(diversity_number)

            # 计算每个多样性值组的引用量均值（上四分位数和下四分位数的均值）
            quartile_citations_by_diversity = {}
            all_quartile_diversity = []
            all_quartile_citations = []
            for diversity, citations in citation_groups.items():
                if len(citations) < 2:
                    mean_citation = np.mean(citations)  # 直接使用原始数据的均值
                    # print(f"Warning: Not enough data for diversity {diversity}. Mean citation set to original value.")
                else:
                    # 计算四分位数
                    q1 = np.percentile(citations, 25)
                    q3 = np.percentile(citations, 75)

                    # 计算四分位数范围内的数据
                    iqr_citations = [c for c in citations if q1 <= c <= q3]

                    # 计算四分位数范围内的均值
                    if iqr_citations:
                        quartile_mean = np.mean(iqr_citations)
                    else:
                        quartile_mean = np.mean(citations)  # 使用原始数据的均值
                        # 如果四分位数均值为 nan，则计算原始数据的均值
                    mean_citation = quartile_mean
                quartile_citations_by_diversity[diversity] = mean_citation
                all_quartile_diversity.append(diversity)
                all_quartile_citations.append(mean_citation)

            # 根据多样性值将数据分组
            for diversity, citation in zip(all_quartile_diversity, all_quartile_citations):
                if 1 <= diversity <= 3:
                    grouped_diversity['low'].append(diversity)
                    grouped_median_citations['low'].append(citation)
                elif 4 <= diversity <= 6:
                    grouped_diversity['medium'].append(diversity)
                    grouped_median_citations['medium'].append(citation)
                elif diversity >= 7:
                    grouped_diversity['high'].append(diversity)
                    grouped_median_citations['high'].append(citation)


            # 打印每组的引用量中位数
            def print_median_citations(grouped_citations):
                for group in grouped_citations:
                    if grouped_citations[group]:
                        median_citation = np.median(grouped_citations[group])
                        print(f"{group.capitalize()} Diversity Group Median Citations: {median_citation:.3f}")


            # print_median_citations(grouped_median_citations)

            # 计算所有数据的皮尔逊相关系数
            if len(all_quartile_diversity) >= 2 and len(all_quartile_citations) >= 2:
                all_quartile_diversity_tensor = tensor(all_quartile_diversity, dtype=torch.float32).to(device)
                all_quartile_citations_tensor = tensor(all_quartile_citations, dtype=torch.float32).to(device)
                pearson_corr = torch.corrcoef(torch.stack((all_quartile_diversity_tensor, all_quartile_citations_tensor)))[
                    0, 1].item()
                print(f"全部的皮尔逊相关系数：{pearson_corr:.3f}")
            else:
                pearson_corr = None
                print("样本数量不足，无法计算全部的皮尔逊相关系数。")

            # 计算每组数据的皮尔逊相关系数
            pearson_corr_by_group = {}
            for group in ['low', 'medium', 'high']:
                if len(grouped_diversity[group]) < 2 or len(grouped_median_citations[group]) < 2:
                    # print(f"{group.capitalize()} Diversity Group does not have enough data for Pearson correlation.")
                    pearson_corr_by_group[group] = None
                    continue

                X = np.array(grouped_diversity[group]).reshape(-1, 1)
                y = np.array(grouped_median_citations[group])

                if np.std(X) == 0 or np.std(y) == 0:
                    # print(f"{group.capitalize()} Diversity Group has zero variance, Pearson correlation is not defined.")
                    pearson_corr_by_group[group] = None
                    continue

                X_tensor = tensor(X, dtype=torch.float32).to(device)
                y_tensor = tensor(y, dtype=torch.float32).to(device)
                pearson_corr_group = torch.corrcoef(torch.stack((X_tensor.squeeze(), y_tensor)))[0, 1].item()
                pearson_corr_by_group[group] = pearson_corr_group
                # print(f"{group.capitalize()} Diversity Group Pearson Correlation: {pearson_corr_group:.3f}")

            # 绘制散点图
            median_diversity = list(quartile_citations_by_diversity.keys())
            median_values = list(quartile_citations_by_diversity.values())
            plt.scatter(median_diversity, median_values, color='black', s=50, zorder=5, label='中位数引用量')

            plt.xlabel('多样性')
            plt.ylabel('引用量')
            plt.title('所有多样性值与对应的中位数引用量图')
            plt.legend()
            # plt.show()

            # 将结果添加到结果字典中
            result = {
                "diversity_number": diversity_number,
                "quartile_citations_by_diversity": quartile_citations_by_diversity,
                "pearson_corr": pearson_corr,
                "pearson_corr_by_group": pearson_corr_by_group
            }

            # filename_key = f"filename_{i}"
            # print(filename_key)
            # 将每个结果逐个写入 JSON 文件
            with open('results_mean3.json', 'a', encoding='utf-8') as f:
                json.dump({filename: result}, f, ensure_ascii=False, indent=4)
                f.write(",\n")  # 每个结果写入后换行

