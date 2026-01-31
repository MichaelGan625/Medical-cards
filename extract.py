import zipfile
import sqlite3
import json
import os
import shutil
import tempfile
from bs4 import BeautifulSoup
import re
# --- 配置区域 ---
ANKI_FILE_PATH = 'AnKing V11 updated.apkg'  # 请修改为你的文件名
OUTPUT_FILE = 'sample.json'
SAMPLE_SIZE = 10000000  # 导师要求先做小批量测试
FILTER_TAG = "Step2"  # 过滤关键词，如果不填则导出所有


def deep_clean_text(text):
    if not text:
        return ""

    # 1. 去除 HTML 标签 (保留空格以免单词粘连)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # 2. 去除音频/图片引用 [sound:...] 或 [img:...]
    text = re.sub(r'\[.*?\]', '', text)

    # 3. 清洗填空题语法 {{c1::Answer}} -> Answer
    # 逻辑：找到 {{c数字::内容}}，只保留内容
    text = re.sub(r'\{\{c\d+::(.*?)(::.*?)?\}\}', r'\1', text)

    # 4. 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_anki_data(apkg_path, output_path, limit=100):
    print(f"🔄 开始处理: {apkg_path}...")

    # 创建临时文件夹来解压数据库
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. 解压 .apkg (本质是 zip)
            with zipfile.ZipFile(apkg_path, 'r') as z:
                z.extractall(temp_dir)

            db_path = os.path.join(temp_dir, 'collection.anki2')

            if not os.path.exists(db_path):
                print("❌ 错误: 无法在包中找到数据库文件 collection.anki2")
                return

            # 2. 连接 SQLite 数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 3. 查询 notes 表 (flds 包含内容, tags 包含标签)
            # Anki 的字段是用 \x1f (Unit Separator) 分隔的字符串
            cursor.execute("SELECT flds, tags FROM notes")

            extracted_data = []
            count = 0
            total_scanned = 0

            print("🔎 正在扫描并清洗数据...")

            for row in cursor:
                flds_str, tags_str = row
                total_scanned += 1

                # 检查标签是否包含 "Step 2" (如果设置了过滤)
                if FILTER_TAG and (FILTER_TAG.lower() not in tags_str.lower()):
                    continue

                # 4. 数据清洗与分割
                fields = flds_str.split('\x1f')

                # 通常 Field 0 是正面(问题), Field 1 是背面(答案)，但也可能有更多字段
                # 我们把所有字段清洗后存入列表
                cleaned_fields = [deep_clean_text(f) for f in fields]

                card_obj = {
                    "id": total_scanned,
                    "tags": tags_str.strip(),
                    "front": cleaned_fields[0] if len(cleaned_fields) > 0 else "",
                    "back": cleaned_fields[1] if len(cleaned_fields) > 1 else "",
                    "extra_fields": cleaned_fields[2:]  # 如果有额外字段
                }

                extracted_data.append(card_obj)
                count += 1

                # 达到限制数量停止
                if count >= limit:
                    break

            conn.close()

            # 5. 保存为 JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)

            print(f"✅ 成功! 已提取 {count} 条包含 '{FILTER_TAG}' 的数据。")
            print(f"📂 结果已保存至: {output_path}")
            print(f"📊 扫描进度: 扫描了前 {total_scanned} 条数据找到目标样本。")

        except Exception as e:
            print(f"❌ 发生错误: {e}")


# --- 运行脚本 ---
if __name__ == "__main__":
    # 确保这里的文件名是正确的
    if not os.path.exists(ANKI_FILE_PATH):
        print(f"⚠️ 找不到文件: {ANKI_FILE_PATH}，请修改脚本中的文件名。")
    else:
        extract_anki_data(ANKI_FILE_PATH, OUTPUT_FILE, limit=SAMPLE_SIZE)