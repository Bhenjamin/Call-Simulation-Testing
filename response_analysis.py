from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen:
    def __init__(self, model_name, persona_file):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Instructions for AI - read from a file
        with open(f"./personas/{persona_file}") as file:
            self.system_instructions = file.read()

        self.history = []

    def generate_response(self, user_input):
        # Conversation context - add user response to conversation history
        conversation_history = self.history + [{"role": "user", "content": user_input}]

        # Initialises some shit idk
        text = self.tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generates the response
        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history - with AIs response 
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response