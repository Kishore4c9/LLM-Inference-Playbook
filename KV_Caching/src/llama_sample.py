import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# 1. Load the tokenizer and model
# Note: Ensure you have run `huggingface-cli login` first
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Initialize chat history
chat_history = [
    {"role": "system", "content": "You are a helpful, coherent AI assistant."}
]

print("--- Llama 3.1 Chat Started (type 'exit' to stop) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    chat_history.append({"role": "user", "content": user_input})

    # 2. Apply the Llama 3.1 Chat Template
    # This automatically adds the <|begin_of_text|>, <|start_header_id|>, etc.
    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 3. Define Stopping Criteria
    # Llama 3.1 needs to know to stop at both the standard EOS and the EOT tag
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 4. Generate
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
        # eos_token_id=tokenizer.eos_token_id
    )

    # 5. Decode only the NEW part
    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output_tokens[0][prompt_len:], skip_special_tokens=True)

    print(f"\nLlama 3.1: {response.strip()}\n")

    # Save the response back to history
    chat_history.append({"role": "assistant", "content": response.strip()})