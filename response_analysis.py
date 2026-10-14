import time
from ollama import chat
from ollama import ChatResponse

MAX_TOKENS = 50 # can use up to 32768 per message  

class ResponseAnalysis:
    def __init__(self, model_name: str, persona_file: str):
        self.model = model_name;

        # Instructions for AI - read from a file
        with open(f"./personas/{persona_file}") as file:
            self.system_instructions = file.read()

        # Add the initial system prompt (persona of the caller)
        # History stores the entire conversation for the AI to review each time
        # a new message is generated 
        self.history = [{"role": "system", "content": self.system_instructions}]

    def generate_response(self, user_input)->str:

        # Conversation context - user response to conversation history
        self.history.append({"role": "user", "content": user_input})

        response: ChatResponse = chat(
            model=self.model,
            messages=self.history,
            think=False,
            options={"num_predict": MAX_TOKENS},
            keep_alive="10m"
        )
        end_time = time.time()

        ai_response = response.message.content or ""

        # Update the conversation history
        self.history.append({"role": "assistant", "content": ai_response})

        print(f"AI Caller: {ai_response}")
        return ai_response