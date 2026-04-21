"""
Simple CLI Chatbot using Hugging Face's Serverless Inference API

Description:
------------
This script creates a command-line chatbot using Hugging Face's hosted LLMs.
It continuously takes user input, sends it to a selected model (Qwen2.5-7B-Instruct),
and prints the model's response.

Features:
---------
- Interactive chat loop
- Uses Hugging Face InferenceClient
"""

import os
from huggingface_hub import InferenceClient


# Initialize the Hugging Face Inference Client
## NOTE: Replace with your actual HF API token or use environment variable
client = InferenceClient(    
    api_key=os.environ["HF_TOKEN"],                   # Recommended way
    # api_key="************",    # Replace with your real huggingface access token
)

print("Chatbot started! Type 'exit' or 'end' to stop.\n")

# Main chat loop
while True:
    # Take user input
    user_input = input("You: ") # Ex: "What is the capital of France?"
    if not user_input.strip():
        print("┌──────────────────────────────────┐")
        print("│  Oops! You didn't type anything. │")
        print("└──────────────────────────────────┘\n")
        continue
    
    # Exit condition
    if user_input.lower() in ('end', 'exit'):
        print("\nBot: Goodbye! Have a great day \n")
        break
        
    
    # Send user input to the model 
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-1M:featherless-ai",
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )

    # Extract and print model respons
    response = completion.choices[0].message['content']
    print(f"Bot: {response}\n")
    
