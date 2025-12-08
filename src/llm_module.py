import ollama

LLM_MODEL_NAME = "mistral"

class LLMClient:
    def __init__(self, system_prompt=None):
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        print("LLM Client initialized.")

    def get_response(self, user_prompt):
        print(f"Sending to LLM: '{user_prompt}'")
        self.history.append({"role": "user", "content": user_prompt})
        try:
            response_stream = ollama.chat(model=LLM_MODEL_NAME, messages=self.history, stream=True)
            full_response = ""
            print("LLM Response: ", end="", flush=True)
            for chunk in response_stream:
                part = chunk['message']['content']
                print(part, end="", flush=True)
                full_response += part
            print()
            self.history.append({"role": "assistant", "content": full_response})
            return full_response
        except Exception as e:
            print(f"\n[ERROR] Could not connect to Ollama server: {e}")
            self.history.pop()
            return "I'm sorry, I'm having trouble connecting to my brain."