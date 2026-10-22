from pathlib import Path
import argparse
from ultralytics import YOLO


def train_model(data_yaml, epochs=30, imgsz=640, batch=8):
    """Fine-tune YOLO on an annotated traffic sign dataset."""

    if not Path(data_yaml).is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=8,
        seed=42,
        project="outputs",
        name="traffic_signs",
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a YOLO traffic sign detector"
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
