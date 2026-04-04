import cv2
from ultralytics import YOLO
from tqdm import tqdm

video_path = "night_drive_taipei.mp4"

# Load model
model = YOLO("DETECT/detect_enlightenGAN_notune/comparison/enlightenGAN_yolov8l2/weights/enlightenGAN_best.pt")
# model.to("cuda")  # optional

# Open video (to get metadata)
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

# YOLO stream inference
results = model.predict(
    source=video_path,
    stream=True,
    imgsz=512
)

# Progress bar
pbar = tqdm(total=total_frames)

for r in results:
    frame = r.plot()  # annotated frame
    out.write(frame)
    pbar.update(1)

pbar.close()
out.release()
cap.release()