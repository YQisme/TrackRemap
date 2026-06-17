import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS_DIR = ROOT / "dataset" / "labels" / "train"
TRACK_ID_DIR = ROOT / "dataset" / "labels_track_id" / "train"
CATEGORY_YAML = ROOT / "category.yaml"
DATA_YAML = ROOT / "dataset" / "data.yaml"


def load_category_entries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    entries = []
    for key, items in data.items():
        merged = {"key": key}
        for item in items:
            merged.update(item)
        entries.append(merged)
    return entries


def load_track_id_to_class(entries: list[dict]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for entry in entries:
        cls = int(entry["classes"])
        for tid in entry.get("track_id") or []:
            mapping[int(tid)] = cls
    return mapping


def load_class_names(entries: list[dict]) -> dict[int, str]:
    return {int(entry["classes"]): entry["key"] for entry in entries}


def load_default_class(entries: list[dict]) -> int:
    for entry in entries:
        if entry["key"] == "noworkwear":
            return int(entry["classes"])
    return max(int(e["classes"]) for e in entries)


def bbox_key(parts: list[str]) -> tuple:
    return tuple(parts[1:5])


def write_data_yaml(names: dict[int, str]):
    path_val = "."
    train_val = "images/train"
    val_val = None
    if DATA_YAML.exists():
        with open(DATA_YAML, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        path_val = existing.get("path", path_val)
        train_val = existing.get("train", train_val)
        val_val = existing.get("val")

    lines = [f"path: {path_val}", f"train: {train_val}"]
    if val_val is not None:
        lines.append(f"val: {val_val}")
    lines.append("")
    lines.append("names:")
    for cls_id in sorted(names):
        lines.append(f"  {cls_id}: {names[cls_id]}")

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_labels():
    entries = load_category_entries(CATEGORY_YAML)
    track_id_to_class = load_track_id_to_class(entries)
    class_names = load_class_names(entries)
    default_class = load_default_class(entries)
    label_files = sorted(LABELS_DIR.glob("*.txt"))

    stats = {"files": 0, "mapped": 0, "defaulted": 0}

    for lbl_path in label_files:
        track_path = TRACK_ID_DIR / lbl_path.name
        if not track_path.exists():
            print(f"跳过（无 track_id 文件）: {lbl_path.name}")
            continue

        with open(lbl_path, encoding="utf-8") as f:
            label_lines = [ln.strip() for ln in f if ln.strip()]
        with open(track_path, encoding="utf-8") as f:
            track_lines = [ln.strip() for ln in f if ln.strip()]

        # 按 bbox 坐标建立 track_id 行索引
        track_by_bbox: dict[tuple, int] = {}
        for line in track_lines:
            parts = line.split()
            if len(parts) < 6:
                continue
            track_by_bbox[bbox_key(parts)] = int(parts[5])

        new_lines = []
        for line in label_lines:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == 0:
                cls = 100

            if cls == 100:
                key = bbox_key(parts)
                tid = track_by_bbox.get(key)
                if tid is not None and tid in track_id_to_class:
                    cls = track_id_to_class[tid]
                    stats["mapped"] += 1
                else:
                    cls = default_class
                    stats["defaulted"] += 1

            new_lines.append(f"{cls} {' '.join(parts[1:])}")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

        stats["files"] += 1

    write_data_yaml(class_names)

    print(f"处理完成: {stats['files']} 个文件")
    print(f"  按 track_id 映射: {stats['mapped']} 行")
    print(f"  默认改为类别 {default_class}: {stats['defaulted']} 行")
    print(f"track_id 映射表: {track_id_to_class}")
    print(f"已更新 data.yaml names: {class_names}")


if __name__ == "__main__":
    process_labels()
