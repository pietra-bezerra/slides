import cv2
import time
import os
import json
from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# Ajuste os caminhos corretamente para o modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'captures')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

app = Flask(__name__)
CORS(app)

# Definições do YOLO
IMG_SIZE = 640
CONF_THRESHOLD = 0.60
COLORS = {
    'Oculos EPI': (0, 255, 0),       # Verde
    'Oculos Comum': (0, 165, 255),   # Laranja
    'Sem Oculos': (0, 0, 255)        # Vermelho
}
CLASSES_MAP = {
    'oculos_epi': 'Oculos EPI',
    'oculos_comum': 'Oculos Comum',
    'sem_oculos': 'Sem Oculos'
}

class Detector:
    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model = None

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap:
            print("Erro: Câmera não encontrada.")
            
        self.last_save_time = 0
        self.history_file = os.path.join(os.path.dirname(__file__), 'history.json')
        self.full_history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_history(self):
        self.full_history = self.full_history[:1000]
        with open(self.history_file, 'w') as f:
            json.dump(self.full_history, f)

    def generate_frames(self):
        while True:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                time.sleep(1) # Espera 1 segundo antes de tentar ler
                
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    time.sleep(1)
                    continue
            else:
                time.sleep(1)
                continue

            # Redimensionar via software para a IA processar
            frame = cv2.resize(frame, (640, 480))

            if self.model:
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
                        
                        # Coloca o texto em cima do quadrado
                        cv2.putText(frame, f"{label} {conf*100:.0f}%", (x1, max(y1 - 10, 0)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        detections_found = True
                        if can_save:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            # Remove acentos para salvar no arquivo
                            clean_label = label.replace('Ó', 'O').replace(' ', '_').lower()
                            filename = f"demo_{clean_label}_seg{conf*100:.0f}_{timestamp}.jpg"
                            filepath = os.path.join(OUTPUT_DIR, filename)
                            cv2.imwrite(filepath, frame)
                            
                            new_entries.append({
                                "tipo": label.replace('_', ' ').title(),
                                "data": time.strftime("%d/%m/%Y"),
                                "hora": time.strftime("%H:%M:%S"),
                                "confianca": f"{conf*100:.0f}%",
                                "imagem": filename
                            })

                if detections_found and can_save:
                    self.last_save_time = current_time
                    for entry in new_entries:
                        self.full_history.insert(0, entry)
                    self._save_history()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

detector = Detector()

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    return jsonify(detector._load_history())

@app.route('/captures/<path:filename>')
def serve_capture(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/clear_history')
def clear_history():
    detector.full_history = []
    detector._save_history()
    # Remove files
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.jpg'):
            try: os.remove(os.path.join(OUTPUT_DIR, f))
            except: pass
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
