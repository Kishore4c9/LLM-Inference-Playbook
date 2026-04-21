from src.chat_kv_cache import ChatKVCache


def main():
    chat_llm = ChatKVCache()

    print("ð Local LLM Chat (KV cache enabled). Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
    
        reply = chat_llm.chat_step(user_input, max_new_tokens=200)
        print(f"LLM: {reply}\n")


if __name__ == "__main__":
    main()