import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体，以便在图表中正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

file_path = '../DIV_results/DIV_results_median_5.json'

# 指定要查找的文件名和对应的组名
file_names_mapping = {
    "chi_2012_1.json": "Citation network model ",
    "chi_2012_2.json": "Combined citation network model",
    "chi_2012_3.json": "Semantic-enhanced citation network model",
    "chi_2012_4.json": "Combined-enhanced citation network model",
    "chi_2012_5.json": "Semantic-combined-enhanced citation network model ",
    "chi_2012_6.json": "Combined-semantic-enhanced citation network model"
}

# 创建一个空字典来存储提取的数据
data_to_plot = {}

# 读取JSON文件并提取所需的数据
with open(file_path, 'r') as f:
    data = json.load(f)
    for item in data:
        for name, content in item.items():
            if name in file_names_mapping.keys():
                Mean_citations = content['median_citations_by_diversity']
                group_name = file_names_mapping[name]  # 使用映射的组名
                data_to_plot[group_name] = Mean_citations  # 存储数据

# 分组计算均值
average_citations = {}

for group_name, citations in data_to_plot.items():
    diversity_groups = {
        "Low diversity": [],
        "Medium diversity": [],
        "High diversity": []
    }

    for diversity_level, citation in citations.items():
        diversity_level = int(diversity_level)
        if 1 <= diversity_level <= 3:
            diversity_groups["Low diversity"].append(citation)
        elif 4 <= diversity_level <= 6:
            diversity_groups["Medium diversity"].append(citation)
        elif diversity_level >= 7:
            diversity_groups["High diversity"].append(citation)

    # 计算每个组的均值并存储
    average_citations[group_name] = {key: np.mean(value) if value else 0 for key, value in diversity_groups.items()}

# 准备绘图数据
plt.figure(figsize=(12, 6))

colors = ['#dcdcdc', '#a0a0a0', '#505050']  # 浅灰, 中灰, 深灰



# 定义填充样式：斜线填充、横线填充、网格填充
hatch_patterns = [' ', '/', '\\']  # 斜线、反斜线、竖线

# 设置柱状图的宽度
bar_width = 0.3

# 绘制数据
x_positions = np.arange(len(file_names_mapping))*1.15 # 模型数量的x轴位置
for i, group_name in enumerate(file_names_mapping.values()):
    # 取出低中高的平均值，并依次绘制紧凑柱状图
    means = [average_citations[group_name].get("Low diversity", 0),
             average_citations[group_name].get("Medium diversity", 0),
             average_citations[group_name].get("High diversity", 0)]
    positions = x_positions[i] + np.array([-bar_width, 0, bar_width])  # 每个模型的3个位置紧密排列
    for j, mean_value in enumerate(means):
        plt.bar(positions[j], mean_value, width=bar_width, color=colors[j], edgecolor='black',
                hatch=hatch_patterns[j], label="Low diversity" if j == 0 and i == 0 else "Medium diversity" if j == 1 and i == 0 else "High diversity" if j == 2 and i == 0 else "")

# 添加标题和标签
plt.title('Average number of citations in the diversity group for each model')
plt.ylabel('Average number of citations')

# 调整 x 轴标签，确保与每组柱状图居中对齐
shortened_labels = [
    "Citation network\nmodel", "Combined citation\nnetwork model", "Semantic-enhanced\ncitation network\nmodel",
    "Combined-enhanced\ncitation network\nmodel", "Semantic-combined-\nenhanced citation\nnetwork model", "Combined-semantic-\nenhanced citation\nnetwork model"
]
plt.xticks(x_positions, shortened_labels, rotation=0, ha='center')  # 使用 x_positions 确保标签居中对齐

# 增加网格线
plt.grid(axis='y')

# 添加图例
plt.legend(title='Diversity group', bbox_to_anchor=(1.05, 1), loc='upper left')

# 调整图表布局
plt.tight_layout()

# 显示图表
plt.show()
