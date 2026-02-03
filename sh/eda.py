import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup


# --- 1. 工具函数 ---
def clean_html(html_str):
    """提取纯文本并移除 Anki 特有语法"""
    if not html_str or not isinstance(html_str, str): return ""
    # 移除完形填空 {{c1::内容}}
    text = re.sub(r'\{\{c\d+::(.*?)\}\}', r'\1', html_str)
    # 移除 HTML 标签
    return BeautifulSoup(text, "html.parser").get_text(separator=' ').strip()


def extract_point(text):
    """通过正则关键词匹配听诊部位"""
    text = text.lower()
    mapping = {
        'Apex/Mitral': ['apex', 'mitral', '5th intercostal'],
        'Aortic': ['aortic', '2nd right'],
        'Pulmonic': ['pulmonic', '2nd left'],
        'Tricuspid': ['tricuspid', '4th left', 'sternal border'],
        'Base': ['base of the heart']
    }
    for k, v in mapping.items():
        if any(word in text for word in v): return k
    return "Other"


# --- 2. 主处理流程 ---
def run_refined_eda(json_file):
    print("正在解析数据并提取维度...")
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    rows = []
    for item in raw_data:
        fields = item.get('patient_data', [])
        if len(fields) < 2: continue

        front_text = clean_html(fields[0])
        back_text = clean_html(fields[1])
        tags = "|".join(item.get('tags', [])).lower()

        # 维度 1: 标签领域分类
        category = "Other/General"
        if "physio" in tags:
            category = "Physiology"
        elif "patho" in tags:
            category = "Pathology"
        elif "pharma" in tags:
            category = "Pharmacology"
        elif "anatomy" in tags:
            category = "Anatomy"

        rows.append({
            "Category": category,
            "WordCount": len(front_text) + len(back_text),
            "Point": extract_point(front_text)
        })
z    df = pd.DataFrame(rows)

    # --- 3. 可视化绘制 ---
    # 设置绘图风格和画布
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # A. 知识领域分布 (条形图)
    sns.countplot(data=df, x='Category', ax=axes[0], palette='coolwarm', order=df['Category'].value_counts().index)
    axes[0].set_title('1. Distribution of knowledge domains', fontsize=14)
    axes[0].set_xlabel('domain')
    axes[0].set_ylabel('amount')

    # B. 卡片字数分布 (密度图)
    sns.histplot(df['WordCount'], bins=50, ax=axes[1], kde=True, color='purple')
    # 限制 x 轴范围以排除极少数超长文本带来的视觉压缩 (取 98 分位数)
    axes[1].set_xlim(0, df['WordCount'].quantile(0.98))
    axes[1].set_title('2. word count length', fontsize=14)
    axes[1].set_xlabel('length')

    # C. 核心听诊部位分布 
    point_counts = df[df['Point'] != "Other"]['Point'].value_counts()
    if not point_counts.empty:
        point_counts.plot.pie(ax=axes[2], autopct='%1.1f%%', colors=sns.color_palette('viridis'), startangle=140)
        axes[2].set_ylabel('')  # 移除纵轴标签
        axes[2].set_title('3.Percentage of core auscultation sites ', fontsize=14)
    else:
        axes[2].text(0.5, 0.5, '未检测到部位特征', ha='center')

    plt.tight_layout()
    plt.savefig('medical_knowledge_eda.png', dpi=300)
    print("分析报告已生成: medical_knowledge_eda.png")
    return df


if __name__ == "__main__":
    df_result = run_refined_eda('extracted_data.json')