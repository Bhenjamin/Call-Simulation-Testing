import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOKENS = 100 # can use up to 32768 per message  

class Qwen:
    def __init__(self, model_name: str, persona_file: str):
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        # Instructions for AI - read from a file
        with open(f"./personas/{persona_file}") as file:
            self.system_instructions = file.read()

        # Add the initial system prompt (persona of the caller)
        # History stores the entire conversation for the AI to review each time
        # a new message is generated 
        self.history = [{"role": "system", "content": self.system_instructions}]
        end_time = time.time()
        print(f"Initilisation Time: {(end_time - start_time)}")

    def generate_response(self, user_input):
        # Conversation context - add system instructions and user response to conversation history
        conversation_history = self.history + [{"role": "user", "content": user_input}]

        # Template used for each chat message
        text = self.tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )

        # Generates the response
        start_time = time.time()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        end_time = time.time()
        print(f"Tokenisation Time: {(end_time - start_time)}")

        start_time = time.time()
        response_ids = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS)[0][len(inputs.input_ids[0]):].tolist()
        end_time = time.time()
        print(f"Generation Time: {(end_time - start_time)}")

        start_time = time.time()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        end_time = time.time()
        print(f"Decording Time: {(end_time - start_time)}")

        # Update history - with AIs response 
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response