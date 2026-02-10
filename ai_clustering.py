import json
import pandas as pd
import torch
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from umap import UMAP

# === 配置区域 ===
INPUT_FILE = 'sample.json'  # 确保文件名正确 (可以是 sample_100.json 测试)
OUTPUT_HTML = 'ai_tier_cluster_map.html'  # 输出文件名

# === 1. 定义 AI 的候选标签 (Category) ===
# 这些是 AI 能听懂的医学词汇
CANDIDATE_LABELS = [
    "Emergency Medicine", "Cardiology", "Neurology", "Respiratory",
    "Gastroenterology", "Pediatrics", "ObGyn", "Infectious Disease",
    "Endocrinology", "Psychiatry", "General Surgery", "Nephrology",
    "Hematology", "Orthopedics", "Dermatology", "Internal Medicine",
    "Family Medicine"
]

# === 2. 定义 Category -> Tier 的映射规则 ===
# AI 算出左边的 Category 后，我们自动把它归入右边的 Tier
TIER_MAPPING = {
    # === Tier S ===
    "Emergency Medicine": "Tier S",
    "Cardiology": "Tier S",
    "Neurology": "Tier S",
    "Respiratory": "Tier S",
    "Gastroenterology": "Tier S",
    "Pediatrics": "Tier S",

    # === Tier A+ ===
    "ObGyn": "Tier A+",
    "Infectious Disease": "Tier A+",
    "Endocrinology": "Tier A+",
    "Psychiatry": "Tier A+",
    "General Surgery": "Tier A+",

    # === Tier A ===
    "Nephrology": "Tier A",
    "Hematology": "Tier A",
    "Orthopedics": "Tier A",
    "Dermatology": "Tier A",

    # === General/Other ===
    "Internal Medicine": "General",
    "Family Medicine": "General"
}


def run_ai_tier_visualization():
    print(f"🚀 Loading Data: {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 如果想先跑小样本测试，可以解开下一行的注释
    # data = data[:100]
    print(f"📊 Processing {len(data)} cards...")

    # 1. 准备文本
    texts = []
    ids = []
    hover_texts = []

    for item in data:
        full_text = f"{item['front']} {item['back']}"
        texts.append(full_text)
        ids.append(item.get('id', 'N/A'))

        display_text = f"Q: {item['front']}<br>A: {item['back']}"
        hover_texts.append(display_text[:300] + "..." if len(display_text) > 300 else display_text)

    # 2. 加载医学模型
    print("📥 Loading Medical AI Model (PubMedBERT)...")
    # 依然使用这个强大的医学模型来做 Embeddings 和分类
    model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

    # 3. 生成 Embeddings (这一步决定图的形状)
    print("🧠 Encoding Cards (Generating Medical Map)...")
    card_embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    # 4. AI 分类 (这一步决定点的颜色)
    print("🤖 Classifying cards into Categories & Tiers...")
    label_embeddings = model.encode(CANDIDATE_LABELS, convert_to_tensor=True)

    # 计算相似度
    cos_scores = util.cos_sim(card_embeddings, label_embeddings)

    # 找出分数最高的 Category
    top_results = torch.argmax(cos_scores, dim=1)

    # 获取具体的分类名称列表
    predicted_categories = [CANDIDATE_LABELS[i] for i in top_results.cpu().numpy()]

    # === 关键步骤：根据 AI 的分类结果，查找对应的 Tier ===
    predicted_tiers = [TIER_MAPPING.get(cat, "Other") for cat in predicted_categories]

    # 5. 降维 (UMAP)
    print("🗺️ Running UMAP dimensionality reduction...")
    umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    proj_2d = umap_2d.fit_transform(card_embeddings.cpu().numpy())

    # 6. 绘图
    print("🎨 Generating Plot...")
    df = pd.DataFrame({
        'x': proj_2d[:, 0],
        'y': proj_2d[:, 1],
        'Category': predicted_categories,  # AI 预测的详细科室
        'Tier': predicted_tiers,  # 映射出来的 Tier S/A+/A
        'Content': hover_texts,
        'ID': ids
    })

    # 排序：让 Tier S 排在前面，图例更好看
    df = df.sort_values(by=['Tier', 'Category'])

    fig = px.scatter(
        df, x='x', y='y',
        color='Category',  # 颜色区分具体科室 (如 Cardiology)
        symbol='Tier',  # 形状区分 Tier (如 S 用圆圈, A 用方块) - 类似你之前的逻辑
        hover_data={'Content': True, 'x': False, 'y': False, 'Category': True, 'Tier': True},
        title='Medical Semantics Clusters (AI Classified + Tiered)',
        template='plotly_white',
        width=1200, height=800
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))

    fig.write_html(OUTPUT_HTML)
    print(f"✅ Success! Plot saved to: {OUTPUT_HTML}")
    print("这张图是完美的结合体：")
    print("1. 坐标(位置)由医学 AI 决定 -> 聚类更科学")
    print("2. 颜色/形状由 Tier 系统决定 -> 符合你的原始分类逻辑")


if __name__ == "__main__":
    run_ai_tier_visualization()