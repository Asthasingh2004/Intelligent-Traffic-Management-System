from flask import Flask, render_template, request, jsonify, Response
import threading
import os
import cv2

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)
lane_data = {
    1: {"count": 0, "signal": "red",  "ambulance": False, "green_time": 0},
    2: {"count": 0, "signal": "red",  "ambulance": False, "green_time": 0},
    3: {"count": 0, "signal": "red",  "ambulance": False, "green_time": 0},
    4: {"count": 0, "signal": "red",  "ambulance": False, "green_time": 0},
}

video_paths   = {1: None, 2: None, 3: None, 4: None}
latest_frames = {1: None, 2: None, 3: None, 4: None}
processing_active = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/upload', methods=['POST'])
def upload_videos():
    global video_paths, processing_active, latest_frames

    # Reset everything for a fresh run
    processing_active = False
    latest_frames = {1: None, 2: None, 3: None, 4: None}
    for i in range(1, 5):
        lane_data[i] = {"count": 0, "signal": "red", "ambulance": False, "green_time": 0}

    uploaded_any = False
    for i in range(1, 5):
        file = request.files.get(f'lane{i}')
        if file and file.filename != '':
            path = os.path.join('uploads', f'lane{i}.mp4')
            file.save(path)
            video_paths[i] = path
            print(f"Uploaded: lane{i}.mp4")
            uploaded_any = True
        else:
            # Use existing file if already present and no new upload
            path = os.path.join('uploads', f'lane{i}.mp4')
            if os.path.exists(path):
                video_paths[i] = path
                print(f"Using existing: lane{i}.mp4")

    if uploaded_any or any(v is not None for v in video_paths.values()):
        processing_active = True
        thread = threading.Thread(target=start_processing)
        thread.daemon = True
        thread.start()

    return ('', 302, {'Location': '/dashboard'})

@app.route('/status_api')
def status_api():
    return jsonify(lane_data)

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    return Response(
        generate_frames(lane_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_frames(lane_id):
    import time
    while True:
        frame = latest_frames.get(lane_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
        time.sleep(0.033)

def start_processing():
    from detector import process_all_lanes
    process_all_lanes(video_paths, lane_data, latest_frames)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)