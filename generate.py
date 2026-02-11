import json
import random

INPUT_FILE = 'sample.json'
OUTPUT_FILE = 'pilot_prompts.txt'
SAMPLE_SIZE = 5


def generate_pilot_prompts():
    print(f"🔄 Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 sample.json，请先运行 extract.py")
        return

    if len(data) > SAMPLE_SIZE:
        selected_cards = random.sample(data, SAMPLE_SIZE)
    else:
        selected_cards = data

    print(f"🎲 Selected {len(selected_cards)} random cards for the pilot study.\n")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=== EXPERIMENT PROMPTS (COPY & PASTE TO LLMS) ===\n\n")

        for card in selected_cards:
            # === 新版 Prompt: "Zero-Fluff Association" (极致对应版) ===
            # === 核心修改：保留 Zero-Fluff 风格，但强制展开口诀 ===
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
            f.write(prompt + "\n\n" + "=" * 50 + "\n\n")

    print(f"✅ Prompts generated in {OUTPUT_FILE}")
    print("👉 Next Step: Open 'pilot_prompts.txt', copy the prompts, and feed them to Gemini, GPT, and Claude.")


if __name__ == "__main__":
    generate_pilot_prompts()
