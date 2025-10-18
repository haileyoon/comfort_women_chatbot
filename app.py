from flask import Flask, request, jsonify, render_template, session
from src.rag import chatbot_response
import os
from dotenv import load_dotenv
from src.tools import get_response

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'response': 'No question provided.'})
    # Retrieve chat history from session
    chat_history = session.get('chat_history', [])
    chat_history.append({"role": "user", "content": question})
    # Call chatbot_response with chat history
    response_text = chatbot_response(question, chat_history)
    chat_history.append({"role": "assistant", "content": response_text})
    session['chat_history'] = chat_history
    MAX_HISTORY = 20
    chat_history = chat_history[-MAX_HISTORY:]
    session['chat_history'] = chat_history
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
