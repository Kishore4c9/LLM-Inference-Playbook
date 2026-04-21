import torch
from transformers import Gemma3ForCausalLM, AutoProcessor

model_id = "google/gemma-3-270m-it"

# 1. Load the processor and model
# This model is small enough to run on almost any CPU or GPU
processor = AutoProcessor.from_pretrained(model_id)
model = Gemma3ForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16 # Use float32 if your hardware doesn't support bfloat16
)

# 2. MANUALLY SET THE TEMPLATE
# This is the official Jinja2 template for Gemma models
processor.chat_template = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\n' }}"
    "{% endif %}"
)

# INITIALIZE HISTORY: To keep track of the conversation
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    # reply = chat_llm.chat_step(user_input, max_new_tokens=200)
    # print(f"LLM: {reply}\n")

    # 3. Prepare the input
    # Gemma 3 uses a specific header format for its chat template
    messages = [
        {"role": "user", "content": user_input}
    ]

    # Append new user message to history
    chat_history.append({"role": "user", "content": user_input})

    inputs = processor.apply_chat_template(
        chat_history, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)

    # 4. Generate
    # output_tokens = model.generate(
    #     **inputs, 
    #     max_new_tokens=128,
    #     do_sample=True,
    #     temperature=0.7
    #)

    # 4. Generate with "Anti-Loop" parameters
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        # repetition_penalty=1.2, # KEY: Stops the model from repeating commas/words
        pad_token_id=processor.pad_token_id,
        eos_token_id=processor.eos_token_id
    )

    # 5. Decode (skipping the prompt tokens)
    prompt_len = inputs.input_ids.shape[1]
    response = processor.decode(output_tokens[0][prompt_len:], skip_special_tokens=True)

    print(f"Gemma: {response.strip()}")
    # Append model response to history so it remembers next time
    chat_history.append({"role": "model", "content": response.strip()})
    