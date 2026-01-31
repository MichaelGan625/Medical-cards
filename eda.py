import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

JSON_FILE = 'large_sample.json'

SUBJECT_MAP = {
    'Cardiology': ['Cardio', 'Heart', 'Vascular'],
    'Neurology': ['Neuro', 'Brain'],
    'Gastroenterology': ['Gastro', 'GI', 'Liver'],
    'Respiratory': ['Respiratory', 'Pulm', 'Lung'],
    'Nephrology': ['Renal', 'Nephro', 'Kidney'],
    'Endocrinology': ['Endo', 'Diabetes', 'Thyroid'],
    'Hematology/Oncology': ['Heme', 'Onco', 'Blood'],
    'Infectious Disease': ['Infectious', 'Micro', 'Bacteria', 'Virus'],
    'Psychiatry': ['Psych', 'Behavioral'],
    'Dermatology': ['Derm', 'Skin'],
    'Musculoskeletal': ['Musculo', 'Ortho', 'Rheum'],
    'Pediatrics': ['Peds', 'Pediatric'],
    'ObGyn': ['ObGyn', 'Obstetrics', 'Gynecology'],
    'Surgery': ['Surgery', 'Surg'],
    'Emergency': ['Emergency', 'EM', 'Trauma'],
    'Internal Medicine': ['IM', 'Internal'],
    'Family Medicine': ['FM', 'Family Medicine', '4fm'],
    'Genetics': ['Genetics'],
}

def extract_medical_category(tag_input):
    if isinstance(tag_input, list):
        tag_str = " ".join(tag_input).lower()
    else:
        tag_str = str(tag_input).lower()

    for category, keywords in SUBJECT_MAP.items():
        for k in keywords:
            k_lower = k.lower()
            if len(k_lower) <= 3:
                pattern = r'(?:^|[^a-z])' + re.escape(k_lower) + r'(?:$|[^a-z])'
                if re.search(pattern, tag_str):
                    return category
            else:
                if k_lower in tag_str:
                    return category
    return "Other"

def run_eda(json_file):
    print(f"📊 正在读取数据并生成最终报告: {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"❌ 读取出错: {e}")
        return

    if 'front' in df.columns:
        df['front_len'] = df['front'].apply(lambda x: len(str(x).split()))
    else:
        df['front_len'] = 0
    if 'back' in df.columns:
        df['back_len'] = df['back'].apply(lambda x: len(str(x).split()))
    else:
        df['back_len'] = 0

    df['Subject'] = df['tags'].apply(extract_medical_category)

    subject_counts = df['Subject'].value_counts()
    if 'Other' in subject_counts:
        other_count = subject_counts['Other']
        subject_counts = subject_counts.drop('Other')
    else:
        other_count = 0

    top_subjects = subject_counts.head(12)

    print("\n" + "=" * 40)
    print("🏆 最终排名 (Top Subjects)")
    print("=" * 40)
    print(top_subjects)
    print(f"\n(注: 有 {other_count} 张卡片归类为 Other/General)")

    try:
        sns.set_style("whitegrid")
    except:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Step 2 Dataset Analysis (N={len(df)}) - Corrected Classification', fontsize=16)

    axes[0].hist(df['front_len'], bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0].set_title('Question Length')
    axes[1].hist(df['back_len'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1].set_title('Answer Length')

    y_pos = range(len(top_subjects))
    palette = sns.color_palette("husl", len(top_subjects))
    axes[2].barh(y_pos, top_subjects.values, color=palette, edgecolor='black', alpha=0.8)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(top_subjects.index, fontsize=11)
    axes[2].invert_yaxis()
    axes[2].set_title('Clinical Subject Distribution')

    for i, v in enumerate(top_subjects.values):
        axes[2].text(v + 10, i, str(v), va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('eda_report.png', dpi=400)
    print(f"\n✅ 成功！修复版图表已保存: eda_report.png")

if __name__ == "__main__":
    run_eda(JSON_FILE)