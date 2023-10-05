from gpt4all import GPT4All
import logging

# Initialize GPT4All model
model_path = "/Users/noni/Library/Application Support/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"
gpt4all_instance = GPT4All(model_path)

system_prompt = "You are a help assistant to assist a master student in Robotics."
prompt_template = "{0}"
gpt4all_instance.chat_session(system_prompt=system_prompt, prompt_template=prompt_template)

logging.basicConfig(level=logging.INFO)

# Initialize messages and custom prompt template
MESSAGES = []
custom_prompt = ""

# Function to handle chat interactions
def chat_loop():
    MESSAGES.clear()
    print("Welcome to the simplified GPT4All CLI!")

    with gpt4all_instance.chat_session('You are a geography expert.\nBe terse.',
                        '### Instruction:\n{0}\n### Response:\n'):
        while True:
            # Custom prompt before user input
            user_input = input(custom_prompt + "User: ")
            
            # Check for quit command
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
              
                # Generate model's response
            full_response = gpt4all_instance.generate(
                # prompt=context_info1 + context_info2 + context_messages +"\n---------------------------------------\n\nThis is What i speak to you this turn of conversation: "+ user_input,
                prompt=user_input,
                max_tokens=100,
                temp=0.7,
                top_k=40,
                top_p=0.4,
                repeat_penalty=1.18,
                repeat_last_n=64,
                n_batch=8
            )
            
            # Extract and print assistant's response
            # assistant_response = full_response.strip().split("\n")[-1]
            assistant_response = full_response
            print("Assistant: " + assistant_response)

if __name__ == "__main__":
    chat_loop()
