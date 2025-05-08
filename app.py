from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import os
from datetime import datetime

# Flask app initialization
app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Global context
session_context = {}

# File paths
MODEL_FILE = "data.pth"
LOG_FILE = 'logs.json'
UNKNOWN_FILE = 'unknown_queries.json'


# Load the trained model and metadata
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model file not found. Train the model first.")
    
    data = torch.load(MODEL_FILE)
    model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
    model.load_state_dict(data["model_state"])
    model.eval()
    
    return model, data["all_words"], data["tags"]

model, all_words, tags = load_model()


# Get chatbot response
def get_response(msg, session_id="default"):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    output = model(X)
    probs = torch.softmax(output, dim=1)
    threshold = 0.90

    high_conf_tags = [
        (tags[i], probs[0][i].item())
        for i in range(len(tags))
        if probs[0][i].item() > threshold
    ]
    high_conf_tags.sort(key=lambda x: x[1], reverse=True)

    responses = []
    for tag, _ in high_conf_tags:
        intent = next((i for i in intents["intents"] if i["tag"] == tag), None)
        if intent:
            responses.append(random.choice(intent["responses"]))

    if responses:
        combined_response = " ".join(responses)
        session_context[session_id] = {"last_tag": high_conf_tags[0][0]}
        log_interaction(msg, combined_response)
        return combined_response

    fallback = get_fallback_response()
    log_interaction(msg, fallback)
    log_unknown_query(msg)
    return fallback


def get_fallback_response():
    fallback_intent = next((i for i in intents["intents"] if i["tag"] == 'fallback'), None)
    return random.choice(fallback_intent["responses"]) if fallback_intent else "I'm sorry, I didn't understand that."


# Log interaction to a file
def log_interaction(message, response):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": message,
        "bot_response": response
    }
    append_json_log(LOG_FILE, log_entry)


# Log unknown query
def log_unknown_query(message):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": message
    }
    append_json_log(UNKNOWN_FILE, log_entry)


# Append entry to a JSON file
def append_json_log(file_path, entry):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, 'w') as f:
                json.dump([entry], f, indent=4)
        else:
            with open(file_path, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error logging to {file_path}: {e}")


# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data.get("message", "")
    response = get_response(msg)
    return jsonify({"reply": response})


# Run app
if __name__ == "__main__":
    app.run(debug=True)
