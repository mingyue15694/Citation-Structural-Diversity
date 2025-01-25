import os
import json

# 读取embedding文件，提取paper_id到paper_ids_1
embedding_file_path = 'F:/mingyue/specter/pythonProject1/Interdisciplinary_computing/Hum_Brain_Mapp_embeddings_1.json'

paper_ids_1 = set()  # 使用集合来存储paper_id，查找更高效
with open(embedding_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        paper_id = data["paper_id"]
        paper_ids_1.add(paper_id)

# 读取PubMed文章数据文件，提取paper_id和References到paper_ids_2
paper_ids_2 = set()  # 使用集合来存储paper_id和引用的paper_id
file_path = 'F:/mingyue/specter/pythonProject1/Interdisciplinary_computing/Hum_Brain_Mapp_articles.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for entry in data:
        paper_id = entry['PMID']
        paper_ids_2.add(paper_id)
        references = entry.get('References', [])
        for ref_id in references:
            paper_ids_2.add(ref_id)

# 找出paper_ids_2中但paper_ids_1中没有的paper_id
unique_in_paper_ids_2 = paper_ids_2 - paper_ids_1
print(len(paper_ids_1))
print(len(paper_ids_2))
# 打印结果
print(f"Number of unique paper_ids in paper_ids_2 but not in paper_ids_1: {len(unique_in_paper_ids_2)}")
# print(f"List of unique paper_ids: {list(unique_in_paper_ids_2)}")
