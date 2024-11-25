import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box

WEIGHT = './yolov7-tiny.pt'
input_video_path = '../Lab_08_Test.mp4'
output_video_path = '../output_video.mp4'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video {input_video_path}")
    exit()
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_orig = frame.copy()
    image = letterbox(frame, (640, 640), stride=64, auto=True)[0]
    if device == "cuda":
        image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    else:
        image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)
    with torch.no_grad():
        output = model(image)[0]
    output = non_max_suppression_kpt(output, 0.25, 0.65)[0]

    if output is not None and len(output) > 0:
        output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
        for *xyxy, conf, cls in output:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=2)

    out.write(image_orig)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Detection completed. Output video saved to {output_video_path}")
