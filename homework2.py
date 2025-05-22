import torch
import cv2
import numpy as np

# 모델 로드
model_path = "yolov5s.pt"
model = torch.hub.load('.', 'custom', path=model_path, source='local')
model.eval()
names = model.names

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)  # 웹캠 연결 (0번 장치)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    window_name = "YOLOv5 Webcam Detection"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Confidence", window_name, 25, 100, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        conf_thres = cv2.getTrackbarPos("Confidence", window_name) / 100.0

        results = model(frame, size=640)
        detections = results.xyxy[0]

        for *xyxy, conf, cls in detections:
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # 창 크기 축소 (50%)
        scale = 0.5
        resized_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        cv2.imshow(window_name, resized_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
