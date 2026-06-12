import cv2
import json
import time
import os
from ultralytics import YOLO

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

# Parâmetros de Performance (Equilíbrio Fluidez/Precisão)
IMG_SIZE = 640   
CONF_THRESHOLD = 0.5 

# Pastas de saída
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'captures')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    print("Iniciando sistema de detecção com Rastreamento Estabilizado...")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # Tentar abrir a câmera em diferentes índices e backends
    cap = None
    for index in [0]:
        print(f"Tentando abrir camera {index} (DSHOW)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Tenta ler um frame para garantir que a câmera funciona
            ret, _ = cap.read()
            if ret: break
            
        print(f"Tentando abrir camera {index} (Padrao)...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: break
    
    if cap is None or not cap.isOpened():
        print("Erro: Nenhuma câmera disponível foi encontrada.")
        return

    # Configurações para mínima latência
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    last_save_time = 0
    
    history_file = 'history.json'
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                full_history = json.load(f)
        except:
            full_history = []
    else:
        full_history = []

    print("Monitoramento ativo. Use 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # Usar model.track com persist=True para manter os quadrados estáveis e sem lag
        # verbose=False remove logs desnecessários para ganhar velocidade
        results = model.track(frame, persist=True, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

        detections_found = False
        new_entries = [] 
        
        current_time = time.time()
        can_save = (current_time - last_save_time) >= 3.0

        for result in results:
            if result.boxes is None: continue
            
            for box in result.boxes:
                # Obter classe e IDs de rastreio
                cls_id = int(box.cls[0])
                label = CLASSES_MAP.get(model.names[cls_id], model.names[cls_id])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                color = COLORS.get(label, (255, 255, 255))

                # Desenhar caixa mais suave e fina para não "poluir" o rosto
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label estilizado
                txt = f"{label} {conf:.2f}"
                cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if can_save:
                    detections_found = True
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    img_name = f"det_{timestamp_str}_{frame_count}.jpg"
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
            last_save_time = current_time 
            
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

        # Exibição a 60fps (ou o máximo da câmera) sem saltos de frames
        cv2.imshow('EPI Guard - Monitoramento', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("Sistema encerrado.")

if __name__ == "__main__":
    main()
