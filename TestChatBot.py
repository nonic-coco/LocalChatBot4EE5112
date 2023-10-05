import json
import random
import typer

from gpt4all import GPT4All

# Initialize chatbot
# gpt4all_instance = GPT4All("ggml-gpt4all-j-v1.3-groovy")  # Replace with your model name
gpt4all_instance = GPT4All("/Users/noni/Library/Application Support/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin")

# Read dataset from jsonl file
dataset_file_path = "./boolq/dev.jsonl"  # Replace with your dataset file path
questions_answers = []

with open(dataset_file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        questions_answers.append((data['question'], data['answer']))

# Randomly select 500 questions with a seed of 216
random.seed(216)
selected_questions_answers = random.sample(questions_answers, 500)

# Function to test chatbot
def test_chatbot(question, expected_answer, results_file):
    MESSAGES = [
        {"role": "system", "content": "You will be given a sentence and you need to judge the sentence by answering 'true' or 'false' before other response."},
        {"role": "user", "content": question}
    ]
    conversation = "\n".join(message["content"] for message in MESSAGES)
    # Generate a response based on the concatenated conversation
    full_response = gpt4all_instance.generate(conversation)

    # Extract the assistant's response from the generated output
    assistant_response = full_response.strip().split("\n")[-1]

    # Your logic to interpret assistant's response and convert it to True/False
    # For demonstration, assuming assistant's response is "True" or "False"
    assistant_answer = assistant_response.strip().lower() == "true"

    is_correct = assistant_answer == expected_answer

    result_data = {
        "question": question,
        "expected_answer": expected_answer,
        "response": str(assistant_response),
        "assistant_answer": assistant_answer,
        "correct": is_correct  # Add whether it's correct or not
    }

    with open(results_file, 'a') as rf:
        json.dump(result_data, rf)
        rf.write('\n')

    return is_correct


# Define the file where you want to store the results
results_file_path = "./boolq/result_chat_gpt4.json"

# Test selected questions
correct_count = 0
total_count = len(selected_questions_answers)

for question, expected_answer in selected_questions_answers:
    if test_chatbot(question, expected_answer, results_file=results_file_path):
        print(f"Correct answer for: {question}")
        correct_count += 1
    else:
        print(f"Incorrect answer for: {question}")

print(f"Accuracy: {correct_count / total_count * 100}%")
