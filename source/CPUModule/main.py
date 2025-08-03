import cv2
from ultralytics import YOLO

# Załaduj gotowy model YOLOv8n (n = nano = szybki)
model = YOLO("yolov8n.pt")  # lub yolov5s.pt dla YOLOv5

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)  # 0 = domyślna kamera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Wykryj obiekty
    results = model(frame)

    # Rysuj wykryte obiekty
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Rysuj ramkę i nazwę klasy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Pokaż wynik
    cv2.imshow("Wykrywanie obiektów", frame)

    # Wyjście po wciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
