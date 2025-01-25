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
                    paper_id = entry['paper_id']
                    references = entry.get('references', [])
                    if paper_id not in G:
                        G.add_node(paper_id)
                    for ref_id in references:
                        if ref_id not in G:
                            G.add_node(ref_id)
                        G.add_edge(paper_id, ref_id)
    return G


data_folder = 'output_date_files3_largest_component'
G = build_citation_network(data_folder)
# 结构多样性值的文件夹
date_folder = '../year_venue_50_div1'

# 引文量的文件夹
venue_folder = 'year_venue_50'

# 初始化用于整个文件夹的多样性值和引用量
all_diversity_values = []
all_venue_values = []

year_data = {}
# 加载发表年份
with open('date_clean_later8.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        year_data[data["paper_id"]] = int(data["year"])

# 遍历结构多样性值文件夹中的所有文件
for filename in os.listdir(date_folder):
    if filename.endswith(".json"):
        # 初始化多样性字典
        print(filename)
        diversity_values = {f"diversity_values_{i}": [] for i in range(1, 2)}
        venue_values = []
        paper_ids = []
        file_path = os.path.join(date_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            paper_id = None
            for line in f:
                # 使用正则表达式匹配键值对
                matches = re.findall(r'"([^"]+)": (\d+)', line)
                if matches:
                    for key, value in matches:
                        if key == "paper_id":
                            paper_id = int(value)
                            break
                    for key, value in matches:
                        if key.startswith("node_diversity"):
                            i = int(key.split("node_diversity")[1])
                            diversity_values[f"diversity_values_{i}"].append({paper_id: int(value)})

        # 读取引用量文件
        data_path = os.path.join(venue_folder, filename)
        paper_ids = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 提取 paper_id
                    match = re.search(r'"paper_id": (\d+)', line)
                    if match:
                        paper_id = int(match.group(1))
                        paper_ids.append(paper_id)

        # 读取其引用量
        for node_id in G:
            if node_id not in paper_ids:
                continue
            out_neighbors = list(G.successors(node_id))
            if len(out_neighbors) < 5:
                continue
            in_degree = sum(1 for neighbor in G.predecessors(node_id) if
                            abs(year_data.get(node_id, 0) - year_data.get(neighbor, 0)) < 3)
            venue_values.append({node_id: in_degree})
        # print(venue_values)

        # 将当前文件的数据添加到总集合中
        all_diversity_values.extend(diversity_values[f"diversity_values_{i}"])
        all_venue_values.extend(venue_values)

        for i in range(1, 2):
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
            # print(diversity_number)

            # 计算每个多样性值组的引用量中位数
            median_citations_by_diversity = {}
            all_median_diversity = []
            all_median_citations = []
            for diversity, citations in citation_groups.items():
                median_citation = np.median(citations)
                median_citations_by_diversity[diversity] = median_citation
                all_median_diversity.append(diversity)
                all_median_citations.append(median_citation)

            # 根据多样性值将数据分组
            for diversity, citation in zip(all_median_diversity, all_median_citations):
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
            if len(all_median_diversity) >= 2 and len(all_median_citations) >= 2:
                all_median_diversity_tensor = tensor(all_median_diversity, dtype=torch.float32).to(device)
                all_median_citations_tensor = tensor(all_median_citations, dtype=torch.float32).to(device)
                pearson_corr = torch.corrcoef(torch.stack((all_median_diversity_tensor, all_median_citations_tensor)))[
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

            # # 绘制散点图
            # median_diversity = list(median_citations_by_diversity.keys())
            # median_values = list(median_citations_by_diversity.values())
            # plt.scatter(median_diversity, median_values, color='black', s=50, zorder=5, label='中位数引用量')
            # 
            # plt.xlabel('多样性')
            # plt.ylabel('引用量')
            # plt.title('所有多样性值与对应的中位数引用量图')
            # plt.legend()
            # # plt.show()

            # 将结果添加到结果字典中
            result = {
                "diversity_number": diversity_number,
                "median_citations_by_diversity": median_citations_by_diversity,
                "pearson_corr": pearson_corr,
                "pearson_corr_by_group": pearson_corr_by_group
            }

            # 将每个结果逐个写入 JSON 文件
            with open('DIV_results_medain1.json', 'a', encoding='utf-8') as f:
                json.dump({filename: result}, f, ensure_ascii=False, indent=4)
                f.write(",\n")  # 每个结果写入后换行

