import os
import json
import networkx as nx
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import torch

# 检查CUDA是否可用，以便在可用时使用GPU进行计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# 调用函数读取JSON文件并构建引文网络结构图
data_folder = 'output_date_files3_largest_component'
G = build_citation_network(data_folder)

# 初始化多样性字典和入度计数字典
diversity_values = []  # 存储结构多样性值的列表
in_degree_counts = []  # 存储每个 paper_id 的入度节点数量

# 数据文件路径
folder_file = 'year_venue_div'
venue_file = 'date_clean_later8.json'

# 逐行读取多样性数据文件，提取paper_id和特定的node_diversity
for filename in os.listdir(folder_file):
    if filename.endswith(".json"):
        paper_ids = []
        file_path = os.path.join(folder_file, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 正则表达式匹配所有键值对
                matches = re.findall(r'"([^"]+)": (\d+)', line)
                if matches:
                    for key, value in matches:
                        if key == "paper_id":
                            paper_id = int(value)  # 提取paper_id
                            break
                    for key, value in matches:
                        if key.startswith("node_diversity"):
                            i = int(key.split("node_diversity")[1])
                            if i == 1:  # 只选择特定的 node_diversity
                                div_value = int(value)
                                diversity_values.append({paper_id: div_value})  # 存储paper_id和多样性值
                            else:
                                continue

# 将结构多样性转化为回归模型的特征和标签
X = []
y = []

for entry in diversity_values:
    p_id = list(entry.keys())[0]
    div_value = list(entry.values())[0]
    find_l = p_id
    with open(venue_file, 'r', encoding='utf-8') as embedding_file:
        for _ in range(find_l - 1):
            next(embedding_file)
        data = json.loads(next(embedding_file))
    year_num = int(data["year"])
    n_citation = int(data["n_citation"])

    out_degree_count = G.out_degree(p_id)
    count2 = 0
    if p_id in G:
        # 获取所有入度节点
        predecessors = list(G.predecessors(p_id))
        count = 0
        for id in predecessors:
            find_line1 = id
            if id > 3509437:
                continue
            with open(venue_file, 'r', encoding='utf-8') as embedding_file1:
                for _ in range(find_line1 - 1):
                    next(embedding_file1)
                data1 = json.loads(next(embedding_file1))
            year = int(data1["year"])

            if abs(year - year_num) <= 5:
                count += 1
            if abs(year - year_num) <= 3:
                count2 += 1

        in_degree_counts.append({p_id: count})

    # 获取对应的入度节点数量，默认为0
    in_degree_count = next((item[p_id] for item in in_degree_counts if p_id in item), 0)

    # 特征是结构多样性值
    X.append([div_value, out_degree_count, count2])

    # 标签是引用量（即入度节点数量）
    y.append(in_degree_count)

X = np.array(X)
y = np.array(y)

# 划分训练集和测试集 (80%训练集，20%测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
lr_model = LinearRegression(n_jobs=1)
lr_model.fit(X_train, y_train)

# 训练KNN回归模型
knn_model = KNeighborsRegressor(
    n_neighbors=7,
    weights='uniform',
    algorithm='auto',
    leaf_size=40,
    p=1,
    metric='minkowski'
)
knn_model.fit(X_train, y_train)

# 训练SVR回归模型（多项式核）
svr_poly_model = SVR(
    kernel='poly',
    C=1000.0,
    degree=2
)
svr_poly_model.fit(X_train, y_train)


# 训练CART回归模型
cart_model = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='random',
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=0
)
cart_model.fit(X_train, y_train)

# 定义并训练 MLP 回归模型
mlp_model = MLPRegressor(
    hidden_layer_sizes=(200,100),
    activation='tanh',
    solver='adam',
    learning_rate='adaptive',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42
)
mlp_model.fit(X_train, y_train)

# 进行预测
lr_predictions = lr_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
svr_poly_predictions = svr_poly_model.predict(X_test)
cart_predictions = cart_model.predict(X_test)
mlp_predictions = mlp_model.predict(X_test)

# 评估模型性能
models = {
    "Linear Regression": lr_predictions,
    "KNN Regression": knn_predictions,
    "SVR (Poly Kernel)": svr_poly_predictions,
    "CART Regression": cart_predictions,
    "MLP Regression": mlp_predictions
}

for name, predictions in models.items():
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"{name}:")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")
