
# from flask import Flask, render_template, request, jsonify, Response
# from transformers import ViTForImageClassification, ViTImageProcessor
# import torch
# from PIL import Image
# import cv2
# import numpy as np
# import tempfile
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Load the saved model and processor
# model_path = "C:/Users/Aakash/Desktop/deefake with UI/webapp/saved_model"
# model = ViTForImageClassification.from_pretrained(model_path)
# processor = ViTImageProcessor.from_pretrained(model_path)
# model.eval()

# # Define labels
# labels = ["Real", "Fake"]

# def process_image(image):
#     """Process an image and return the predicted label and confidence scores."""
#     try:
#         inputs = processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#         logits = outputs.logits
#         predicted_class = torch.argmax(logits, dim=1).item()
#         return labels[predicted_class], logits.softmax(dim=1).tolist()[0]
#     except Exception as e:
#         print("Error during image processing:", str(e))
#         raise

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict_image', methods=['POST'])
# def predict_image():
#     """Handle image upload and return prediction results."""
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']
#     image = Image.open(file.stream).convert("RGB")
#     predicted_label, confidence_scores = process_image(image)

#     return jsonify({
#         "predicted_label": predicted_label,
#         "confidence_scores": confidence_scores
#     })
    

# def generate_video_stream(video_path):
#     """Stream video frames with predictions overlaid."""
#     video_capture = cv2.VideoCapture(video_path)
#     try:
#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 break

#             # Convert frame to PIL Image for processing
#             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             predicted_label, confidence_scores = process_image(image)

#             # Overlay predictions on the frame
#             label_text = f"{predicted_label} (Real: {confidence_scores[0]:.2%}, Fake: {confidence_scores[1]:.2%})"
#             cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             # Encode frame as JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame_bytes = buffer.tobytes()

#             # Yield frame as part of MJPEG stream
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     finally:
#         video_capture.release()

# @app.route('/stream_video', methods=['POST'])
# def stream_video():
#     """Stream video with predictions overlayed."""
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']
#     video_path = tempfile.NamedTemporaryFile(delete=False).name
#     file.save(video_path)

#     return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from transformers import ViTForImageClassification, ViTImageProcessor
import sqlite3
import torch
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key"  # Secret key for session

# Database setup
DATABASE = "users.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load the saved model and processor
model_path = "C:/Users/Sunali/OneDrive/Desktop/webapp_signupLogin/saved_model"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()

# Define labels
labels = ["Real", "Fake"]

def process_image(image):
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return labels[predicted_class], logits.softmax(dim=1).tolist()[0]
    except Exception as e:
        print("Error during image processing:", str(e))
        raise

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Email already registered.")
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user'] = user[0]
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    predicted_label, confidence_scores = process_image(image)

    return jsonify({
        "predicted_label": predicted_label,
        "confidence_scores": confidence_scores
    })


def generate_video_stream(video_path):
    """Stream video frames with predictions overlaid."""
    video_capture = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert frame to PIL Image for processing
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predicted_label, confidence_scores = process_image(image)

            # Overlay predictions on the frame
            label_text = f"{predicted_label} (Real: {confidence_scores[0]:.2%}, Fake: {confidence_scores[1]:.2%})"
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame as part of MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        video_capture.release()

@app.route('/stream_video', methods=['POST'])
def stream_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    video_path = tempfile.NamedTemporaryFile(delete=False).name
    file.save(video_path)

    return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
