import argparse
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, split="test"):
    """Evaluate a trained traffic sign detection model."""
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split=split,
        plots=True,
    )

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the traffic sign detector"
    )
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])

    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.split)
