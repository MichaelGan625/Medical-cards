import json
import os
import time
import concurrent.futures
import threading
from tqdm import tqdm
from openai import OpenAI

# ================= 🚀 全量配置区域 =================
# 1. API 设置
DEEPSEEK_API_KEY = "sk-998d487791454d4394b7054ef08e6dc8"
BASE_URL = "https://api.deepseek.com"

# 建议用 deepseek-chat (V3)，省钱且速度极快
MODEL_NAME = "deepseek-chat"

# 2. 文件路径
INPUT_FILE = 'sample.json'
# 使用 JSONL 确保数据实时落盘，绝对安全
OUTPUT_FILE = 'deepseek_cards_full.jsonl'

# 3. 性能与限制
TEST_LIMIT = 0  # 0 = 跑全量，不限制数量！
MAX_WORKERS = 25  # 提速到 50 线程，榨干 API 性能
# ===============================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
file_lock = threading.Lock()


def construct_prompt(card):
    # =================================================================
    # 🔴 严格使用你指定的 PROMPT，绝不修改
    # =================================================================
    prompt = f"""
        --- START PROMPT FOR CARD ID {card.get('id')} ---

        ## Role
        You are an expert at creating "Zero-Fluff" medical memory hooks. 

        ## The Objective
        The user wants to eliminate ALL unnecessary words (is, was, results in, leads to). 
        Format the card as a direct **Logic Association**: [Trigger Term] ----> {{{{c1::[Target Fact]}}}}.

        ## Strict Formatting Rules
        1. **No Sentences:** Do NOT write full, grammatically correct sentences. 
        2. **Minimal Connection:** Use a colon (:), an arrow (->), or just bold labels.
        3. **Cloze Strategy:** Always cloze the specific fact, value, or treatment. 
        4. **Brevity:** The entire front should ideally be under 10-15 words.

        ## CRITICAL RULE: Mnemonics & Lists (FIX FOR PROFESSOR)
        If the card is about a Mnemonic (like PAIR, TORCH) or a List:
        - **DO NOT** cloze the Mnemonic name itself (e.g., do NOT write {{c1::PAIR}}).
        - **DO** cloze the *meaning* of the mnemonic items.
        - **Format:** **Mnemonic Name** ----> {{{{c1::Item 1, Item 2, Item 3}}}}

        ## Examples of "Perfect" Zero-Fluff Cards

        [Input: Normal HR 60-100]
        {{
          "improved_front": "Normal **Adult Heart Rate**: {{{{c1::60–100 bpm}}}}",
          "improved_back": ""
        }}

        [Input: PAIR stands for Psoriasis, Ankylosing, IBD, Reactive]
        {{
          "improved_front": "**HLA-B27 (PAIR)** ----> {{{{c1::Psoriasis, Ankylosing, IBD, Reactive}}}}",
          "improved_back": "Seronegative Spondyloarthropathies"
        }}

        [Input: Post-op MI treatment is PCI and Heparin]
        {{
          "improved_front": "**Post-op MI** Tx ----> {{{{c1::PCI / Heparin}}}}",
          "improved_back": "Revascularization"
        }}

        ## Raw Data
        Front: {card.get('front')}
        Back: {card.get('back')}

        ## Output Requirement
        Return ONLY a valid JSON object:
        {{
            "card_id": "{card.get('id')}",
            "improved_front": "**Trigger** ----> {{{{c1::Target Fact}}}}",
            "improved_back": "Short Hint (Optional)"
        }}
        --- END PROMPT ---
                    """
    return prompt


def call_deepseek_api(card):
    prompt = construct_prompt(card)
    retries = 3

    while retries > 0:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={'type': 'json_object'}
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)
            result_json['card_id'] = card.get('id')
            return result_json

        except Exception as e:
            # 报错只打印 ID，不刷屏
            print(f"\n❌ [ID: {card.get('id')}] Retry ({3 - retries}/3) - {e}")
            retries -= 1
            time.sleep(2)

    return None


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 断点续传逻辑
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip():
                        data = json.loads(line)
                        processed_ids.add(str(data['card_id']))
                except:
                    pass

    print(f"📂 总任务量: {len(raw_data)} | ✅ 已完成: {len(processed_ids)}")

    tasks = [c for c in raw_data if str(c['id']) not in processed_ids]

    if TEST_LIMIT > 0:
        print(f"🚧 【测试模式】只跑前 {TEST_LIMIT} 条...")
        tasks = tasks[:TEST_LIMIT]

    if not tasks:
        print("✅ 所有数据已全部跑完！无需重复运行。")
        return

    print(f"🚀 全速启动！处理剩余 {len(tasks)} 条数据 (并发: {MAX_WORKERS})...")

    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_card = {executor.submit(call_deepseek_api, card): card for card in tasks}

        # 'a' 模式追加写入，buffering=1 开启行缓冲（极致安全）
        with open(OUTPUT_FILE, 'a', encoding='utf-8-sig', buffering=1) as f_out:
            pbar = tqdm(total=len(tasks), unit="card", smoothing=0.1)

            for future in concurrent.futures.as_completed(future_to_card):
                result = future.result()
                if result:
                    json_line = json.dumps(result, ensure_ascii=False)
                    with file_lock:
                        f_out.write(json_line + "\n")
                        f_out.flush()  # 每一条都强制落盘
                    success_count += 1
                pbar.update(1)
            pbar.close()

    print(f"\n🎉 任务完成！")
    print(f"✅ 新增保存: {success_count} 条")
    print(f"📂 结果文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()