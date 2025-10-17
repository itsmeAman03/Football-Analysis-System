from ultralytics import YOLO

model = YOLO('yolov11l_finetune/weights/best.pt')

result = model.predict('inputs/08fd33_4.mp4', save=True)
print(result[0])
print('='*20)
boxes = result[0].boxes

try:
    boxes = result[0].boxes
    if boxes is not None and hasattr(boxes, 'xyxy'):
        for box in boxes.xyxy:
            print(box)
    else:
        print("No boxes found.")
except Exception as Error:
    print("Error", Error)
