import json
import random

# === 配置 ===
INPUT_FILE = 'sample.json'  # 你的数据源
OUTPUT_FILE = 'pilot_prompts.txt'  # 输出的 Prompt 文本
SAMPLE_SIZE = 5


def generate_pilot_prompts():
    print(f"🔄 Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 sample.json，请先运行 extract.py")
        return

    # 随机抽取
    if len(data) > SAMPLE_SIZE:
        selected_cards = random.sample(data, SAMPLE_SIZE)
    else:
        selected_cards = data

    print(f"🎲 Selected {len(selected_cards)} random cards for the pilot study.\n")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=== EXPERIMENT PROMPTS (COPY & PASTE TO LLMS) ===\n\n")

        for card in selected_cards:
            # === 新版 Prompt: Natural Clinical Flow (自然流畅版) ===
            prompt = f"""
        --- START PROMPT FOR CARD ID {card.get('id')} ---

        ## Role
        You are a medical tutor preparing materials for a student who values **readability** and **intuition**.

        ## The Problem
        Previous cards were too robotic and "dictionary-like" (e.g., "**Disease X** is characterized by **{{{{c1::Symptom Y}}}}**").
        This is hard to read.

        ## The Goal: "Natural Clinical Flow"
        Create **Cloze Deletion** cards that read like **natural, spoken sentences** a doctor would say to a student.
        1. **Keep it Short:** Still under 20 words.
        2. **Make it Intuitive:** Use normal grammar (Subject -> Verb -> Object). Avoid choppy fragments.
        3. **Context is Key:** The sentence should inherently explain *what* we are talking about.

        ## Formatting Rules
        - **improved_front**: A clean, natural sentence with `{{{{c1::answer}}}}`. Do NOT use bolding (`**`) unless absolutely necessary for distinction.
        - **improved_back**: A short "Why" or "Mechanism" to help understanding (The "Aha!" moment).

        ## Examples Comparison

        [Bad - Robotic]
        "**SIBO** is associated with {{{{c1::increased}}}} levels of **Folate**." (Too dry, hard to scan)

        [Good - Natural]
        "Small Intestinal Bacterial Overgrowth (SIBO) typically results in {{{{c1::increased}}}} levels of Folate due to bacterial synthesis." (Better flow, easier to read)

        [Bad - Robotic]
        "**Ebstein anomaly** is characterized by {{{{c1::downward displacement}}}} of tricuspid valve."

        [Good - Natural]
        "Ebstein anomaly is a congenital defect defined by the {{{{c1::downward displacement}}}} of the tricuspid valve leaflets."

        ## Raw Data
        Front: {card.get('front')}
        Back: {card.get('back')}

        ## Output Requirement
        Return ONLY a valid JSON object:
        {{
            "card_id": "{card.get('id')}",
            "improved_front": "Natural sentence with {{{{c1::answer}}}}",
            "improved_back": "Brief explanation of the mechanism or distinct feature"
        }}
        --- END PROMPT ---
                    """
            f.write(prompt + "\n\n" + "=" * 50 + "\n\n")

    print(f"✅ Prompts generated in {OUTPUT_FILE}")
    print("👉 Next Step: Open 'pilot_prompts.txt', copy the prompts, and feed them to Gemini, GPT, and Claude.")


if __name__ == "__main__":
    generate_pilot_prompts()