import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO

def gen_frames(cctv_num):
    model = YOLO("models/yolo11n.pt")
    url = f"rtsp://210.99.70.120:1935/live/cctv{cctv_num}.stream"
    results = model.track(source = url, conf = 0.15, device = "mps", stream_buffer= True, vid_stride = 5, show_labels = False, show_conf = False, stream = True, verbose = False)

    while True:
        result = next(results)
        frame = result.plot()

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cctv/<cctv_num>')
def cctv(cctv_num):
    return render_template('cctv.html')

@app.route('/cctv_feed/<cctv_num>')
def video_feed(cctv_num):
    return Response(gen_frames(cctv_num), mimetype='multipart/x-mixed-replace; boundary=frame',direct_passthrough=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5001, debug = True)
