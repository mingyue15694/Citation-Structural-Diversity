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
def build_citation_network(data_folder1):
    G = nx.DiGraph()  # 创建有向图
    for filename in os.listdir(data_folder1):  # 遍历文件夹中的文件
        if filename.endswith(".json"):  # 只处理 .json 文件
            file_path = os.path.join(data_folder1, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    paper_id = entry['PMID']  # 获取文章 ID
                    try:
                        paper_id = int(paper_id)  # 尝试转换为整数
                    except ValueError:
                        print(f"Invalid paper ID: {paper_id}, skipping.")
                        continue  # 跳过无效的 ID

                    references = entry.get('References', [])  # 获取引用列表
                    if paper_id not in G:  # 如果节点不存在，则添加
                        G.add_node(paper_id)

                    for ref_id in references:
                        try:
                            ref_id = int(ref_id)  # 尝试将引用 ID 转为整数
                            if ref_id not in G:
                                G.add_node(ref_id)
                            G.add_edge(paper_id, ref_id)  # 添加边
                        except ValueError:
                            # print(paper_id)
                            # print(f"Invalid reference ID: {ref_id}, skipping.")
                            continue  # 跳过无效的引用 ID
    return G

data_folder1 = 'F:/mingyue/specter/pythonProject1/Extract_Output_1'
G = build_citation_network(data_folder1)

# 结构多样性值的文件夹
date_folder = 'F:/mingyue/specter/pythonProject1/pubmed_div'

# 引文量的文件夹
venue_folder = 'F:/mingyue/specter/pythonProject1/Extract_Output_1'

year_data = {}
# 遍历结构多样性值文件夹中的所有文件
for filename in os.listdir(venue_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(venue_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        for entry in file_data:
            paper_id = entry.get("PMID", "")
            year = entry.get("ArticleDate", "")
            year_data[paper_id] = int(year)

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
            data = json.load(f)  # 假设文件是 JSON 格式

        # 遍历数据
        for item in data:  # 遍历列表
            for outer_key, inner_data in item.items():
                for inner_key, paper_data in inner_data.items():
                    # 检查 paper_data 是否为列表类型
                    if isinstance(paper_data, list):
                        for data in paper_data:  # 遍历列表
                            paper_id = data.get("paper_id")
                            if paper_id:
                                paper_id = int(paper_id)  # 转换为整数
                                paper_ids.append(paper_id)
                                # 获取 node_diversity 值
                                for key, value in data.items():
                                    if key.startswith("node_diversity"):
                                        try:
                                            i = int(key.split("node_diversity")[1])
                                            if f"diversity_values_{i}" in diversity_values:
                                                diversity_values[f"diversity_values_{i}"].append({paper_id: int(value)})
                                        except ValueError:
                                            pass  # 如果 'node_diversity' 后的数字无法解析为整数，跳过
                    elif isinstance(paper_data, dict):  # 如果是字典
                        paper_id = paper_data.get("paper_id")
                        if paper_id:
                            paper_id = int(paper_id)  # 转换为整数
                            paper_ids.append(paper_id)
                            # 获取 node_diversity 值
                            for key, value in paper_data.items():
                                if key.startswith("node_diversity"):
                                    try:
                                        i = int(key.split("node_diversity")[1])
                                        if f"diversity_values_{i}" in diversity_values:
                                            diversity_values[f"diversity_values_{i}"].append({paper_id: int(value)})
                                    except ValueError:
                                        pass  # 如果 'node_diversity' 后的数字无法解析为整数，跳过

        # print("Diversity Values:", diversity_values)
        # 读取其引用量
        for node_id in G:
            node_id = int(node_id)
            if node_id not in paper_ids:
                continue
            print(node_id)
            in_degree = sum(1 for neighbor in G.predecessors(node_id) if
                            abs(year_data.get(node_id, 0) - year_data.get(neighbor, 0)) < 3)
            venue_values.append({node_id: in_degree})
        # print("*******************************************************************************************")
        # print(venue_values)

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

            # 计算每个多样性值组的引用量的均值
            quartile_citations_by_diversity = {}
            all_median_diversity = []
            all_median_citations = []
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
                all_median_diversity.append(diversity)
                all_median_citations.append(mean_citation)

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
                    print(f"{group.capitalize()} Diversity Group does not have enough data for Pearson correlation.")
                    pearson_corr_by_group[group] = None
                    continue

                X = np.array(grouped_diversity[group]).reshape(-1, 1)
                y = np.array(grouped_median_citations[group])

                if np.std(X) == 0 or np.std(y) == 0:
                    print(
                        f"{group.capitalize()} Diversity Group has zero variance, Pearson correlation is not defined.")
                    pearson_corr_by_group[group] = None
                    continue

                X_tensor = tensor(X, dtype=torch.float32).to(device)
                y_tensor = tensor(y, dtype=torch.float32).to(device)
                pearson_corr_group = torch.corrcoef(torch.stack((X_tensor.squeeze(), y_tensor)))[0, 1].item()
                pearson_corr_by_group[group] = pearson_corr_group

            plt.xlabel('多样性')
            plt.ylabel('引用量')
            plt.title('所有多样性值与对应的中位数引用量图')
            plt.legend()
            # plt.show()

            # 将结果添加到结果字典中
            result = {
                "diversity_number": diversity_number,
                "median_citations_by_diversity": quartile_citations_by_diversity,
                "pearson_corr": pearson_corr,
                "pearson_corr_by_group": pearson_corr_by_group
            }

            # 将每个结果逐个写入 JSON 文件
            with open('DIV_results_4.json', 'a', encoding='utf-8') as f:
                json.dump({filename: result}, f, ensure_ascii=False, indent=4)
                f.write(",\n")  # 每个结果写入后换行
