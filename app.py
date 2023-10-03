import io
import sys
from collections import namedtuple
from gpt4all import GPT4All
from flask import Flask, request, render_template

app = Flask(__name__)

VersionInfo = namedtuple('VersionInfo', ['major', 'minor', 'micro'])
VERSION_INFO = VersionInfo(1, 0, 2)
VERSION = '.'.join(map(str, VERSION_INFO))

class ChatBot:
    def __init__(self, model="/Users/noni/Library/Application Support/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin", n_threads=4):
        self.gpt4all_instance = GPT4All(model)
        if n_threads is not None:
            self.gpt4all_instance.model.set_thread_count(n_threads)

    def chat(self, message):
        response_generator = self.gpt4all_instance.generate(
            message,
            max_tokens=200,
            temp=0.9,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=64,
            n_batch=9,
            streaming=True,
        )
        response = io.StringIO()
        for token in response_generator:
            response.write(token)
        return response.getvalue()

bot = ChatBot()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_message = request.form.get("user_message")
        if user_message:
            assistant_response = bot.chat(user_message)
            return render_template("index.html", user_message=user_message, assistant_response=assistant_response)
    return render_template("index.html", user_message=None, assistant_response=None)

if __name__ == "__main__":
    app.run(debug=True)
