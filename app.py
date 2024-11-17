import os
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/path/to/shape_predictor_68_face_landmarks.dat')

# Function to compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for detecting blink
EAR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 3  # Consecutive frames for blink detection

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400

        # Save the uploaded video
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the video and determine if it's real or deepfake
        is_real = process_video(file_path)

        # Redirect to the result page with the evaluation result
        return redirect(url_for('result', is_real=is_real))
    
    return render_template('upload.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    blink_detected = False  # Flag to track if a blink is detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # If EAR is below the threshold, a blink is detected
            if ear < EAR_THRESHOLD:
                blink_detected = True
                break  # Exit the loop as one blink is enough to classify as real

        if blink_detected:
            break  # Stop processing as soon as a blink is detected

    cap.release()
    return blink_detected

@app.route('/result')
def result():
    is_real = request.args.get('is_real', 'False')  # Default to 'False' if not provided
    is_real = is_real == 'True'  # Convert string to boolean

    # Determine the result based on the presence of a blink
    result_message = "Real Video" if is_real else "Deepfake"
    return render_template('result.html', result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)

