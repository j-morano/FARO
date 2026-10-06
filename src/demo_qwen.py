import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def demo_qwen():
    # 1. Configuration (Extracted from your load_llm function)
    llm_id = "Qwen/Qwen3.5-0.8B-Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {llm_id} on {device}...")

    # 2. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_id)

    # Custom pad token logic from your file
    pad_token = "<|im_end|>"
    tokenizer.pad_token = pad_token

    # 3. Initialize Model
    # Using bfloat16 as specified in your original script
    model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        dtype=torch.bfloat16,
        device_map={"": device},
        tie_word_embeddings=True
    )

    # 4. Generate Text
    prompt = "Augen-OCT Befundbericht: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 5. Decode and Print
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nOutput:\n{output_text}")

if __name__ == "__main__":
    demo_qwen()
