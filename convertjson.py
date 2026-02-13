import json

# 1. 你的输入文件名 (JSONL)
INPUT_FILE = 'llama_cards_safe.jsonl'
# 2. 你的输出文件名 (标准 JSON)
OUTPUT_FILE = 'llama_cards.json'

data = []

print("🔄 正在转换格式...")

# 一行行读取，捏成一个大列表
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip(): # 防止空行报错
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass # 跳过坏掉的行

# 一次性写入
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ 转换完成！已生成标准 JSON 文件: {OUTPUT_FILE}")
print(f"共包含 {len(data)} 条数据。")