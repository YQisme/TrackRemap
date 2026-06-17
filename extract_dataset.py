import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path
from tqdm import tqdm

VIDEO = "a.mp4"
MODEL = "yolo11n.pt"
EXPORT_TRACK_ID = True  # True: 额外导出 labels_track_id 与带 track_id 注释视频

DATASET_ROOT = "dataset"
OUT_IMG = "dataset/images/train"
OUT_LBL = "dataset/labels/train"
OUT_LBL_TRACK_ID = "dataset/labels_track_id/train"
OUT_ID = "dataset/labels_track_id/id"
OUT_VIDEO = "dataset/annotated_track.mp4"
DATA_YAML = "dataset/data.yaml"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)
if EXPORT_TRACK_ID:
    os.makedirs(OUT_LBL_TRACK_ID, exist_ok=True)
    os.makedirs(OUT_ID, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"推理设备: {'GPU (cuda:0)' if DEVICE == 0 else 'CPU'}")

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"无法打开视频: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
save_id = 0
video_writer = None
seen_track_ids = set()
pbar = tqdm(total=total_frames or None, desc="处理视频", unit="帧")


def track_color(track_id: int):
    return (
        int(37 * track_id % 255),
        int(17 * track_id % 255),
        int(29 * track_id % 255),
    )


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.4,
        iou=0.5,
        imgsz=1280,
        device=DEVICE,
        verbose=False
    )[0]

    h, w = frame.shape[:2]

    if EXPORT_TRACK_ID and video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    img_name = f"{save_id:06d}.jpg"
    lbl_name = f"{save_id:06d}.txt"

    cv2.imwrite(os.path.join(OUT_IMG, img_name), frame)

    lines = []
    track_lines = []
    annotated = frame.copy() if EXPORT_TRACK_ID else None

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if EXPORT_TRACK_ID:
                track_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {track_id}")
                if track_id >= 0 and track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    cls_name = model.names[cls]
                    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        id_img = f"{track_id}_{cls_name}.jpg"
                        cv2.imwrite(os.path.join(OUT_ID, id_img), crop)
                color = track_color(track_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"id:{track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

    with open(os.path.join(OUT_LBL, lbl_name), "w") as f:
        f.write("\n".join(lines))

    if EXPORT_TRACK_ID:
        with open(os.path.join(OUT_LBL_TRACK_ID, lbl_name), "w") as f:
            f.write("\n".join(track_lines))
        video_writer.write(annotated)

    save_id += 1
    pbar.update(1)
    pbar.set_postfix(已保存=save_id, 当前人数=len(lines))

pbar.close()
cap.release()
if video_writer is not None:
    video_writer.release()


def write_data_yaml():
    dataset_path = Path(DATASET_ROOT).resolve().as_posix()
    content = f"""path: {dataset_path}
train: images/train
val: images/train

names:
  0: person
"""
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(content)


write_data_yaml()

print(f"✅ DONE: 共处理 {save_id} 帧，数据集已生成")
print(f"📄 配置文件: {Path(DATA_YAML).resolve()}")
if EXPORT_TRACK_ID:
    print(f"🏷️  track_id 标签: {Path(OUT_LBL_TRACK_ID).resolve()}")
    print(f"🆔  track_id 首帧截图: {Path(OUT_ID).resolve()} ({len(seen_track_ids)} 个)")
    print(f"🎬 注释视频: {Path(OUT_VIDEO).resolve()}")
