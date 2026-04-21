import os
from huggingface_hub import InferenceClient

# Initialize the client
# Ensure HF_TOKEN is set in your environment variables
client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],                   # Recommended way
    # api_key="************",    # Replace with your real huggingface access token
)

def start_interactive_chat():
    # 1. Initialize history with an optional system instruction
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    print("--- Chat Session Started (Type 'exit' to quit) ---")

    while True:
        # 2. Capture user input
        user_input = input("\nYou: ")
        
        if not user_input.strip():
            print("┌──────────────────────────────────┐")
            print("│  Oops! You didn't type anything. │")
            print("└──────────────────────────────────┘\n")
            continue
    
        # Exit condition
        if user_input.lower() in ('end', 'exit'):
            print("\nBot: Goodbye! Have a great day \n")
            break

        # 3. Append user message to history
        messages.append({"role": "user", "content": user_input})

        try:
            # 4. Send the entire conversation history to the model
            completion = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct-1M:featherless-ai",
                messages=messages,
                max_tokens=500
            )

            # 5. Extract the assistant's response
            assistant_res = completion.choices[0].message
            response = assistant_res['content']
            
            # 6. Print and append assistant response to history
            
            print(f"Bot: {response}\n")
            messages.append(assistant_res)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_interactive_chat()