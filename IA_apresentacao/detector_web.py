import cv2
import json
import time
import os
import threading
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Configurações de Otimização
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
CLASSES_MAP = {
    'oculos-epi': 'COM EPI',
    'sem_oculos': 'SEM EPI'
}
COLORS = {
    'COM EPI': (0, 255, 0),
    'SEM EPI': (0, 0, 255)
}

IMG_SIZE = 640   
CONF_THRESHOLD = 0.5 

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'captures')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'history.json')

class Detector:
    def __init__(self):
        print("Iniciando sistema de detecção com Rastreamento Estabilizado...")
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            self.model = None

        self.cap = None
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Tentando backend padrão...")
            self.cap = cv2.VideoCapture(0)

        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_count = 0
        self.last_save_time = 0

    def generate_frames(self):
        if not self.cap or not self.cap.isOpened() or not self.model:
            yield b''
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            results = self.model.track(frame, persist=True, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

            detections_found = False
            new_entries = [] 
            current_time = time.time()
            can_save = (current_time - self.last_save_time) >= 3.0

            for result in results:
                if result.boxes is None: continue
                
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = CLASSES_MAP.get(self.model.names[cls_id], self.model.names[cls_id])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    color = COLORS.get(label, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt = f"{label} {conf:.2f}"
                    cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if can_save:
                        detections_found = True
                        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                        img_name = f"det_{timestamp_str}_{self.frame_count}.jpg"
                        img_path = os.path.join(OUTPUT_DIR, img_name)
                        cv2.imwrite(img_path, frame)

                        new_entry = {
                            "class": label,
                            "confidence": conf,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "image": f"captures/{img_name}"
                        }
                        new_entries.append(new_entry)
            
            if detections_found:
                self.last_save_time = current_time 
                if os.path.exists(history_file):
                    try:
                        with open(history_file, 'r') as f:
                            current_history = json.load(f)
                    except:
                        current_history = []
                else:
                    current_history = []

                for entry in new_entries:
                    current_history.insert(0, entry)
                
                current_history = current_history[:1000] 
                with open(history_file, 'w') as f:
                    json.dump(current_history, f)

            # Codificar o frame em JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

detector = Detector()

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

import glob
from flask import send_from_directory

@app.route('/status')
def status():
    return jsonify({"status": "running"})

@app.route('/history')
def get_history():
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return jsonify(json.load(f))
        except:
            return jsonify([])
    return jsonify([])

@app.route('/clear_history', methods=['POST', 'GET'])
def clear_history():
    files = glob.glob(os.path.join(OUTPUT_DIR, '*'))
    for f in files:
        try:
            os.remove(f)
        except:
            pass
    with open(history_file, 'w') as f:
        json.dump([], f)
    return jsonify({"status": "cleared"})

@app.route('/captures/<path:filename>')
def serve_capture(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Roda o servidor Flask na porta 5000 acessível a todas as interfaces
    app.run(host='0.0.0.0', port=5000, threaded=True)
