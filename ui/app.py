from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_for_jarvis!'
socketio = SocketIO(app, async_mode='threading')

# Store current status and message history so newly connected clients can be initialized
current_status = "Starting..."
history = []  # list of dicts: {"sender": "...", "message": "..."}

def update_status(new_status):
    global current_status
    current_status = new_status
    socketio.emit('status_update', {'status': new_status})

def add_message(sender, message):
    # keep small history (e.g., last 200 messages)
    history.append({'sender': sender, 'message': message})
    if len(history) > 200:
        history.pop(0)
    socketio.emit('new_message', {'sender': sender, 'message': message})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('[UI] Client connected')
    # Emit the current status only to the newly connected client
    emit('status_update', {'status': current_status})
    # Send existing history to the newly connected client
    for msg in history:
        emit('new_message', msg)

def run_ui():
    print("[UI] Starting Flask server on http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)