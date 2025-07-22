from flask import Flask, request, send_file, render_template
import os, uuid, cv2, numpy as np
import mediapipe as mp

app = Flask(__name__)
UPLOAD = "uploads"
RESULT = "results"
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULT, exist_ok=True)
mp_pose = mp.solutions.pose

def process(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    pose = mp_pose.Pose()

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            pts = res.pose_landmarks.landmark
            xs = [int(p.x * w) for p in pts]
            ys = [int(p.y * h) for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        out.write(frame)
    cap.release(); out.release()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    vid = request.files['video']
    temp_name = str(uuid.uuid4()) + ".mp4"
    in_path = os.path.join(UPLOAD, temp_name)
    out_path = os.path.join(RESULT, temp_name)
    vid.save(in_path)
    process(in_path, out_path)
    return send_file(out_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
