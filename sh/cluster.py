import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


# --- 1. 数据预处理 ---
def prepare_medical_data(json_file, sample_size=1000):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts, ids = [], []
    for item in data[:sample_size]:  # 采样以保证绘图效率
        fields = item.get('patient_data', [])
        if len(fields) < 2: continue
        clean_t = BeautifulSoup(fields[0] + " " + fields[1], "html.parser").get_text(separator=' ').strip()
        texts.append(clean_t if clean_t else "Empty")
        ids.append(item.get('note_id'))
    return texts, ids


# --- 2. 核心指标计算 
def get_metrics(embeddings):
    # a. 计算余弦相似度矩阵 s_i
    sim_matrix = cosine_similarity(embeddings)

    # b. 计算邻近度 p_i 
    s_c = np.percentile(sim_matrix, 20)
    proximity = np.sum(np.sign(sim_matrix - s_c), axis=1)

    # c. 模拟置信度 c_i 
    confidence = 1.0 - (np.random.rand(len(embeddings)) * 0.5 + (
                1 - (proximity - proximity.min()) / (proximity.max() - proximity.min())) * 0.5)

    return sim_matrix, proximity, confidence


# --- 3. 原型选择与聚类 ---
def perform_clustering(embeddings, sim_matrix, proximity, confidence, n_clusters=6):
    # 选择困难原型: 低置信度 + 高邻近度
    diff_idx = np.argsort(confidence)[:n_clusters // 2]
    d_protos = [diff_idx[np.argmax(proximity[diff_idx])]]

    # 选择异常原型: 最低邻近度
    a_protos = np.argsort(proximity)[:n_clusters // 2].tolist()

    all_protos = d_protos + a_protos
    # 计算每个样本到原型的平均相似度 
    avg_sim_to_protos = np.mean(sim_matrix[:, all_protos], axis=1)

    # 聚类分配
    clusters = np.argmax(cosine_similarity(embeddings, embeddings[all_protos]), axis=1)
    return clusters, all_protos, avg_sim_to_protos


# --- 4. 绘图函数 ---
def visualize_results(df, embeddings, protos):
    fig = plt.figure(figsize=(20, 8))

    # 图 1: 3D 指标空间 (相似度-邻近度-置信度) 
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(df['Proximity'], df['Similarity'], df['Confidence'],
                          c=df['Cluster'], cmap='viridis', alpha=0.6, s=20)

    # 高亮原型点
    ax1.scatter(df.loc[protos, 'Proximity'], df.loc[protos, 'Similarity'], df.loc[protos, 'Confidence'],
                c='red', marker='X', s=200, label='Prototypes')

    ax1.set_xlabel('Proximity (Density)')
    ax1.set_ylabel('Similarity (to Protos)')
    ax1.set_zlabel('Confidence (Model)')
    ax1.set_title('Protoformer 3D Feature Space')
    ax1.legend()

    # 图 2: 2D PCA 降维投影 - 展示簇的边界
    ax2 = fig.add_subplot(122)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(embeddings)

    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=df['Cluster'],
                    palette='viridis', ax=ax2, alpha=0.5, legend='full')
    plt.scatter(pca_res[protos, 0], pca_res[protos, 1], c='red', marker='X', s=150, label='Centers')
    ax2.set_title('PCA Projection of Clusters')

    plt.tight_layout()
    plt.savefig('protoformer_visualization.png')
    print("可视化完成：图表已保存至 protoformer_visualization.png")


# --- 执行 ---
texts, n_ids = prepare_medical_data('extracted_data.json')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

sim, prox, conf = get_metrics(embeddings)
clusters, protos, avg_sim = perform_clustering(embeddings, sim, prox, conf)

df_res = pd.DataFrame({
    'Proximity': prox, 'Similarity': avg_sim, 'Confidence': conf, 'Cluster': clusters
})

visualize_results(df_res, embeddings, protos)