import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os
import networkx as nx
from torch import tensor


# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 提取主题的文件
topic_folder = 'E:/Interdisciplinary_computing/Extract_Topic'

# 结构多样性值的文件
div_folder = 'Hum_Brain_Mapp_div_new_1.json'

# 初始化多样性值和主题数量
diversity_values = {f"diversity_values_{i}": [] for i in range(1, 2)}
diversity_number = []
topic_number = {}
paper_ids = []

# 读取结构多样性值
with open(div_folder, 'r', encoding='utf-8') as f:
    paper_id = None
    for line in f:
        # 使用正则表达式匹配键值对
        matches = re.findall(r'"([^"]+)":\s*([^,\s]+)', line)
        if matches:
            # print(matches)
            for key, value in matches:
                if key == "paper_id":
                    # 去掉字符串两侧的引号
                    paper_id = int(value.strip('"'))
                    # print(paper_id)
                    if paper_id not in paper_ids:
                        paper_ids.append(paper_id)
                    break  # 找到 paper_id 后可以停止遍历
            for key, value in matches:
                if key.startswith("node_diversity"):
                    i = int(key.split("node_diversity")[1])
                    # print(i)
                    diversity_values[f"diversity_values_{i}"].append({paper_id: int(value)})


# 读取主要主题的数量
for filename in os.listdir(topic_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(topic_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # 提取所有文章的id和主题数量
        for data in data_list:
            paper_id = data.get("PMID", "")
            paper_id = int(paper_id)
            MeSH = data.get("MeSHHeadings", [])
            # 统计 #text 的数量
            text_count = sum(1 for mesh in MeSH if "#text:" in mesh)
            topic_number[paper_id] = text_count

# print(diversity_values)
# print(topic_number)

# 计算多样性值与主题数量的相关性
all_diversity = []
all_topics = []


for i in range(1, 2):
    all_diversity = []
    all_topics = []
    grouped_diversity = {'low': [], 'medium': [], 'high': []}
    grouped_median_topics = {'low': [], 'medium': [], 'high': []}
    diversity_number = {'low': 0, 'medium': 0, 'high': 0}
    key = f"diversity_values_{i}"
    diversity_list = diversity_values[key]
    diversity_groups = {}
    topic_groups = {}
    topic_dict = topic_number


    # 将多样性值与主题数量对应起来
    for diversity_value in diversity_list:
        paper_id = list(diversity_value.keys())[0]
        diversity1 = list(diversity_value.values())[0]
        topic_count = topic_dict.get(paper_id, None)
        # print(topic_count)
        if topic_count is not None:
            all_diversity.append(diversity1)
            all_topics.append(topic_count)
            if diversity1 in diversity_groups:
                diversity_groups[diversity1].append(paper_id)
                topic_groups[diversity1].append(topic_count)
            else:
                diversity_groups[diversity1] = [paper_id]
                topic_groups[diversity1] = [topic_count]

    # 统计每个多样性组的文献数量
    for diversity_group in diversity_groups:
        if 1 <= diversity_group <= 3:
            diversity_number['low'] += len(diversity_groups[diversity_group])
        elif 4 <= diversity_group <= 6:
            diversity_number['medium'] += len(diversity_groups[diversity_group])
        elif diversity_group >= 7:
            diversity_number['high'] += len(diversity_groups[diversity_group])
    # print(diversity_number)

    # 计算每个多样性值组的主题数量均值（上四分位数和下四分位数的均值）
    quartile_topics_by_diversity = {}
    all_quartile_diversity = []
    all_quartile_topics = []
    for diversity, topics in topic_groups.items():
        if len(topics) < 2:
            mean_topic = np.mean(topics)  # 直接使用原始数据的均值
            # print(f"Warning: Not enough data for diversity {diversity}. Mean topic set to original value.")
        else:
            # 计算四分位数
            q1 = np.percentile(topics, 25)
            q3 = np.percentile(topics, 75)

            # 计算四分位数范围内的数据
            iqr_topics = [c for c in topics if q1 <= c <= q3]

            # 计算四分位数范围内的均值
            if iqr_topics:
                quartile_mean = np.mean(iqr_topics)
            else:
                quartile_mean = np.mean(topics)  # 使用原始数据的均值
                # 如果四分位数均值为 nan，则计算原始数据的均值
            mean_topic = quartile_mean
        quartile_topics_by_diversity[diversity] = mean_topic
        all_quartile_diversity.append(diversity)
        all_quartile_topics.append(mean_topic)

    # 根据多样性值将数据分组
    for diversity, topic in zip(all_quartile_diversity, all_quartile_topics):
        if 1 <= diversity <= 3:
            grouped_diversity['low'].append(diversity)
            grouped_median_topics['low'].append(topic)
        elif 4 <= diversity <= 6:
            grouped_diversity['medium'].append(diversity)
            grouped_median_topics['medium'].append(topic)
        elif diversity >= 7:
            grouped_diversity['high'].append(diversity)
            grouped_median_topics['high'].append(topic)


    # 打印每组的主题数量中位数
    def print_median_topics(grouped_topics):
        for group in grouped_topics:
            if grouped_topics[group]:
                median_topic = np.median(grouped_topics[group])
                print(f"{group.capitalize()} Diversity Group Median topics: {median_topic:.3f}")


    # print_median_topics(grouped_median_topics)

    # 计算所有数据的皮尔逊相关系数
    if len(all_quartile_diversity) >= 2 and len(all_quartile_topics) >= 2:
        all_quartile_diversity_tensor = tensor(all_quartile_diversity, dtype=torch.float32).to(device)
        all_quartile_topics_tensor = tensor(all_quartile_topics, dtype=torch.float32).to(device)
        pearson_corr = torch.corrcoef(torch.stack((all_quartile_diversity_tensor, all_quartile_topics_tensor)))[
            0, 1].item()
        print(f"全部的皮尔逊相关系数：{pearson_corr:.3f}")
    else:
        pearson_corr = None
        print("样本数量不足，无法计算全部的皮尔逊相关系数。")

    # 计算每组数据的皮尔逊相关系数
    pearson_corr_by_group = {}
    for group in ['low', 'medium', 'high']:
        if len(grouped_diversity[group]) < 2 or len(grouped_median_topics[group]) < 2:
            # print(f"{group.capitalize()} Diversity Group does not have enough data for Pearson correlation.")
            pearson_corr_by_group[group] = None
            continue

        X = np.array(grouped_diversity[group]).reshape(-1, 1)
        y = np.array(grouped_median_topics[group])

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
    median_diversity = list(quartile_topics_by_diversity.keys())
    median_values = list(quartile_topics_by_diversity.values())
    plt.scatter(median_diversity, median_values, color='black', s=50, zorder=5, label='中位数主题数量')

    plt.xlabel('多样性')
    plt.ylabel('主题数量')
    plt.title('所有多样性值与对应的中位数主题数量图')
    plt.legend()
    # plt.show()

    # 将结果添加到结果字典中
    result = {
        "diversity_number": diversity_number,
        "quartile_topics_by_diversity": quartile_topics_by_diversity,
        "pearson_corr": pearson_corr,
        "pearson_corr_by_group": pearson_corr_by_group
    }

    # filename_key = f"filename_{i}"
    # print(filename_key)
    # 将结果写入 JSON 文件
    with open('DIV_results_mean1.json', 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        f.write(",\n")  # 每个结果写入后换行
