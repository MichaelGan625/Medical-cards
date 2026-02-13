import json
import os
import time
import concurrent.futures
import threading
from tqdm import tqdm
from openai import OpenAI

# ================= 🛡️ 安全配置区域 =================
# 1. API 设置
# 🔴 您代码里的 Key (OpenRouter)
DEEPSEEK_API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct" # 建议：gemini-3-pro-preview 极贵且不稳定，Flash 2.0 目前免费且最快，逻辑更强

# 2. 文件路径
INPUT_FILE = 'sample.json'
# 🔴 改为 .jsonl 格式 (流水线写入，最安全)
OUTPUT_FILE = 'llama_cards_safe.jsonl'

# 3. 性能与测试
MAX_WORKERS = 25     # 🔴 已改为 10
TEST_LIMIT = 0      # 🔴 已改为 50 (测试用，跑完确认无误后改为 0)
SAVE_INTERVAL = 1    # 实时保存，改不改无所谓，下面逻辑是实时的
# ===================================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
file_lock = threading.Lock() # 文件锁

def construct_prompt(card):
    # =================================================================
    # 🔴 严格保留您提供的 PROMPT (Gemini版本)
    # =================================================================
    return f"""
    ## Role
    You are an expert at creating "Zero-Fluff" medical memory hooks. 

    ## The Objective
    The user wants to eliminate ALL unnecessary words. 
    Format the card as a direct **Logic Association**: [Trigger Term] ----> {{{{c1::[Target Fact]}}}}.

    ## Strict Formatting Rules
    1. **No Sentences:** Do NOT write full sentences. 
    2. **Minimal Connection:** Use a colon (:), an arrow (->), or just bold labels.
    3. **Cloze Strategy:** Always cloze the specific fact, value, or treatment. 
    4. **Brevity:** The entire front should be under 10-15 words.

    ## CRITICAL RULE: Mnemonics & Lists
    If the card is about a Mnemonic or List:
    - **DO NOT** cloze the Mnemonic name (e.g., do NOT write {{c1::PAIR}}).
    - **DO** cloze the *meaning* of the items.
    - **Format:** **Mnemonic Name** ----> {{{{c1::Item 1, Item 2, Item 3}}}}

    ## Raw Data
    Front: {card.get('front')}
    Back: {card.get('back')}

    ## Output Requirement
    Return ONLY a valid JSON object (no markdown formatting, no ```json tags):
    {{
        "card_id": "{card.get('id')}",
        "improved_front": "**Trigger** ----> {{{{c1::Target Fact}}}}",
        "improved_back": "Short Hint (Optional)"
    }}
    """

def call_deepseek_api(card):
    prompt = construct_prompt(card)
    retries = 3

    while retries > 0:
        try:
            # 增加 HTTP Referer 头部 (OpenRouter 建议)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant that outputs JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={'type': 'json_object'},
                extra_headers={
                    "HTTP-Referer": "[https://github.com/MedicalFlashcards](https://github.com/MedicalFlashcards)",
                    "X-Title": "Medical Flashcards"
                }
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)
            result_json['card_id'] = card.get('id') # 强制回填 ID
            return result_json

        except Exception as e:
            # 打印简短报错，方便观察
            print(f"\n❌ [ID: {card.get('id')}] Error: {e}")
            retries -= 1
            time.sleep(2)

    return None

def main():
    # 1. 准备工作
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. 断点续传 (读取 .jsonl)
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

    print(f"📂 [Gemini] 原始数据: {len(raw_data)} | 已完成: {len(processed_ids)}")

    # 3. 筛选任务
    tasks = [c for c in raw_data if str(c['id']) not in processed_ids]

    # 4. 🔴 测试模式限制
    if TEST_LIMIT > 0:
        print(f"🚧 【测试模式】仅处理前 {TEST_LIMIT} 条...")
        tasks = tasks[:TEST_LIMIT]

    if not tasks:
        print("✅ 所有任务已完成！")
        return

    print(f"🚀 [Gemini] 开始运行 (并发: {MAX_WORKERS})...")

    # 5. 执行 + 实时写入
    success_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_card = {executor.submit(call_deepseek_api, card): card for card in tasks}
        
        # 'a' 模式追加，buffering=1 确保每行实时写入
        with open(OUTPUT_FILE, 'a', encoding='utf-8-sig', buffering=1) as f_out: 
            pbar = tqdm(total=len(tasks), unit="card")
            
            for future in concurrent.futures.as_completed(future_to_card):
                result = future.result()
                
                if result:
                    json_line = json.dumps(result, ensure_ascii=False)
                    with file_lock:
                        f_out.write(json_line + "\n")
                        f_out.flush() # 强制落盘
                    success_count += 1
                
                pbar.update(1)
            pbar.close()

    print(f"\n🎉 测试结束！")
    print(f"✅ 成功写入 {success_count} 条数据到: {OUTPUT_FILE}")
    print(f"💡 确认无误后，请将代码中的 TEST_LIMIT = 0 改为跑全量。")

if __name__ == "__main__":
    main()