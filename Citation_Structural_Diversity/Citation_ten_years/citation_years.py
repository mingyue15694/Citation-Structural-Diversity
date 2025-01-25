import os
import json
import networkx as nx
import re
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# 重新计算年份和引用数量
def calculate_year_citation(year_citation):
    year_citation_dict = {}
    for entry in year_citation:
        for year, citation in entry.items():
            if year not in year_citation_dict:
                year_citation_dict[year] = citation
            else:
                year_citation_dict[year] += citation
    return year_citation_dict


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


# 读取JSON文件并构建引文网络结构图
data_folder = 'output_date_files3_largest_component'
G = build_citation_network(data_folder)

# 初始化多样性字典
diversity_values = []
venue_values = []

floder_file = 'year_venue_50_div_no18'
venue_file = 'date_clean_later8.json'

# 初始化文献数量计数器
low_diversity_count = 0
medium_diversity_count = 0
high_diversity_count = 0

for filename in os.listdir(floder_file):
    if filename.endswith(".json"):
        print(filename)
        file_path = os.path.join(floder_file, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            paper_id = None
            for line in f:
                matches = re.findall(r'"([^"]+)": (\d+)', line)
                if matches:
                    for key, value in matches:
                        if key == "paper_id":
                            paper_id = int(value)
                            break
                    for key, value in matches:
                        if key.startswith("node_diversity"):
                            i = int(key.split("node_diversity")[1])
                            if i == 1:
                                div_value = int(value)
                                diversity_values.append({paper_id: div_value})
                                break

# 提取每个 paper_id 对应的入度节点
in_degree_nodes = {}

# 按低、中、高多样性组分别计算皮尔逊相关系数
low_diversity_group = []
medium_diversity_group = []
high_diversity_group = []

for entry in diversity_values:
    print(entry)
    year_citation = [{i: 0} for i in range(1, 11)]
    p_id = list(entry.keys())[0]
    div_value = list(entry.values())[0]
    find_l = p_id
    with open(venue_file, 'r', encoding='utf-8') as embedding_file:
        for _ in range(find_l - 1):
            next(embedding_file)
        data = json.loads(next(embedding_file))
    year_num = int(data["year"])

    if p_id in G:
        predecessors = list(G.predecessors(p_id))
        in_degree_nodes[p_id] = predecessors
        for id in predecessors:
            find_line1 = id
            with open(venue_file, 'r', encoding='utf-8') as embedding_file:
                for _ in range(find_line1 - 1):
                    next(embedding_file)
                data = json.loads(next(embedding_file))
            year = int(data["year"])

            if year - year_num <= 10:  # 修改判断条件
                i = year_num - year  # 计算位置
                if i < len(year_citation):  # 确保索引不超出列表长度
                    for key in year_citation[i].keys():
                        year_citation[i][key] += 1

    year_citation_dict = calculate_year_citation(year_citation)
    cumulative_citations = {}
    cumulative_sum = 0
    for year in sorted(year_citation_dict.keys()):
        cumulative_sum += year_citation_dict[year]
        cumulative_citations[year] = cumulative_sum

    total_citations = cumulative_sum
    if total_citations > 0:
        normalized_citations = {year: citation / total_citations for year, citation in cumulative_citations.items()}
    else:
        normalized_citations = {year: 0 for year in cumulative_citations.keys()}

    years = list(normalized_citations.keys())
    normalized_citation_values = list(normalized_citations.values())

    # 根据多样性值将数据分到对应的组并更新文献数量
    if 1 <= div_value <= 3:
        low_diversity_group.append((years, normalized_citation_values))
        low_diversity_count += 1
    elif 4 <= div_value <= 6:
        medium_diversity_group.append((years, normalized_citation_values))
        medium_diversity_count += 1
    else:
        high_diversity_group.append((years, normalized_citation_values))
        high_diversity_count += 1

    # 打印文献数量
print(f"Low Diversity Group: {low_diversity_count} papers")
print(f"Medium Diversity Group: {medium_diversity_count} papers")
print(f"High Diversity Group: {high_diversity_count} papers")


# 计算并打印皮尔逊相关系数
def calculate_pearson(group, group_name):
    all_years = []
    all_citations = []

    for entry in group:
        years, citations = entry
        if len(years) == len(citations) and len(years) > 1:
            all_years.extend(years)
            all_citations.extend(citations)

    if len(all_years) > 1 and len(all_citations) > 1:
        pearson_corr, _ = pearsonr(all_years, all_citations)
        print(f"{group_name} Diversity Group Pearson Correlation Coefficient: {pearson_corr:.3f}")
    else:
        print(f"{group_name} Diversity Group Pearson Correlation Coefficient: 无法计算 (样本数量不足或所有值相同)")


 # 计算每个组的皮尔逊相关系数
calculate_pearson(low_diversity_group, 'Low')
calculate_pearson(medium_diversity_group, 'Medium')
calculate_pearson(high_diversity_group, 'High')


# 绘制每种结构多样性的均值趋势图
def plot_mean_trend(group, label, color):
    if group:
        all_years = [val[0] for val in group]
        min_len = min(len(val) for val in all_years)
        years = all_years[0][:min_len]  # 确保所有年份长度一致
        mean_values = np.mean([val[1][:min_len] for val in group], axis=0)
        plt.plot(years, mean_values, linestyle='-', linewidth=2, color=color, label=label)

plt.figure(figsize=(15, 10))

# 绘制均值趋势线
plot_mean_trend(low_diversity_group, 'Low Diversity Mean Trend', 'blue')
plot_mean_trend(medium_diversity_group, 'Medium Diversity Mean Trend', 'green')
plot_mean_trend(high_diversity_group, 'High Diversity Mean Trend', 'red')

plt.xlabel('Years')
plt.ylabel('Normalized Citations')
plt.title('Mean Trend of Diversity vs. Citations')
plt.legend()
plt.show()
