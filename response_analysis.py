import time
from ollama import chat
from ollama import ChatResponse

MAX_TOKENS = 100 # can use up to 32768 per message  

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

        start_time = time.time()
        response: ChatResponse = chat(
            model=self.model,
            messages=self.history,
            think=False,
            options={"num_predict": MAX_TOKENS}
        )
        end_time = time.time()
        print(f"Prompt Time: {(end_time - start_time)}")

        ai_response = response.message.content or ""

        self.history.append({"role": "assistant", "content": ai_response})

        return ai_response
        