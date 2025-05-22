import torch
import cv2
import numpy as np
import os
import glob

# 모델 로드
model_path = "yolov5s.pt"
model = torch.hub.load('.', 'custom', path=model_path, source='local')
model.eval()

# 클래스 이름 로딩
names = model.names

def choose_image(folder="/home/KTH/work_yolov5/yolov5/data/images"):
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))

    if not image_files:
        print("이미지가 없습니다.")
        return None

    print(f"선택된 이미지: {image_files[0]}")
    return image_files[0]

def nothing(x):
    pass

def main():
    image_path = choose_image()
    if not image_path:
        return

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("이미지를 불러올 수 없습니다.")
        return

    window_name = "YOLOv5 Detection"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Confidence", window_name, 25, 100, nothing)  # 초기값 25%

    while True:
        img = orig_img.copy()
        conf_thres = cv2.getTrackbarPos("Confidence", window_name) / 100.0

        results = model(img, size=640)
        detections = results.xyxy[0]  # tensor: (x1, y1, x2, y2, conf, cls)

        for *xyxy, conf, cls in detections:
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # 윈도우 창 크기 축소
        scale = 0.5
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        cv2.imshow(window_name, resized_img)

        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
