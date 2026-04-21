"""
Simple CLI Chatbot using Hugging Face Inference API

Description:
------------
This script creates a command-line chatbot using Hugging Face's hosted LLMs.
It continuously takes user input, sends it to a selected model (Qwen2.5-7B-Instruct),
and prints the model's response.

Features:
---------
- Interactive chat loop
- Uses Hugging Face InferenceClient
- Exption handling and streaming responses for better UX
"""

import os
import sys
from huggingface_hub import InferenceClient
from requests.exceptions import RequestException


# Initialize the Hugging Face Inference Client
## NOTE: Replace with your actual HF API token or use environment variable
client = InferenceClient(    
    api_key=os.environ["HF_TOKEN"],                   # Recommended way
    # api_key="************",    # Replace with your real huggingface access token
)

print("Chatbot started! Type 'exit' or 'end' to stop.\n")

# Main chat loop
while True:
    try:
        user_input = input("You: ")
        if not user_input.strip():
            print("┌──────────────────────────────────┐")
            print("│  Oops! You didn't type anything. │")
            print("└──────────────────────────────────┘\n")
            continue

        # Exit condition
        if user_input.lower() in ("exit", "end"):
            print("\nChatbot: Goodbye! Have a great day \n")
            break

        print("Bot: ", end="", flush=True)

        # STREAMING RESPONSE
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-1M:featherless-ai",
            messages=[{"role": "user", "content": user_input}],
            stream=True,  # Enable streaming
        )

        # Iterate over streamed chunks
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.get("content", "")
                if delta:
                    print(delta, end="", flush=True)
            except (KeyError, IndexError, AttributeError):
                # Skip malformed chunks safely
                continue

        print("\n")

    # Handle network-related errors
    except RequestException as e:
        print("\n️ Network error occurred. Please check your connection.")
        print(f"Details: {e}\n")

    # Handle API-related issues
    except Exception as e:
        print("\n️ An unexpected error occurred while calling the API.")
        print(f"Details: {e}\n")

        # Optional: avoid crashing the loop
        continue