import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data('sample.json')
texts = [f"{item['front']} {item['back']}" for item in data]
ids = [item['id'] for item in data]

# 2. 生成 Embedding (对应论文中的 e(x))
print("正在计算 Embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# 3. 计算 邻近度 (Proximity - 对应论文公式 4)
# 简单实现：计算每个点到其他所有点的平均相似度
print("正在计算邻近度指标...")
sim_matrix = cosine_similarity(embeddings)
# p_i 的简化版：平均相似度。相似度越低，说明越离群
proximity_scores = np.mean(sim_matrix, axis=1)

# 4. 聚类与降维
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 5. 保存可视化图片 (不使用 plt.show)
print("正在生成分析图表...")
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
            c=clusters, cmap='viridis', alpha=0.6, edgecolors='w')

# 特别标注出 Card 3 和 4
for i, card_id in enumerate(ids):
    if card_id in [3, 4]:
        plt.annotate(f"Card {card_id}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                     fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

plt.title("Semantic Analysis of Anki Cards")
plt.savefig('cluster_analysis.png', dpi=300) # 直接保存为高清图片
print("分析图已保存至: cluster_analysis.png")

# 6. 将聚类结果和邻近度写回数据，方便后续筛选
print("正在导出详细结果...")
analysis_results = []
for i in range(len(data)):
    result = data[i].copy()
    result['cluster_id'] = int(clusters[i])
    result['proximity_score'] = float(proximity_scores[i])
    analysis_results.append(result)

# 按邻近度从小到大排序，最前面的就是所谓的“异常值”候选
analysis_results.sort(key=lambda x: x['proximity_score'])

with open('analyzed_cards.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=4, ensure_ascii=False)
print("详细数据已保存至: analyzed_cards.json")