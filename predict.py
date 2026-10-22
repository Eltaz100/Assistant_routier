from pathlib import Path
from ultralytics import YOLO


def predict_image(
    image_path: str,
    model_path: str = "models/best.pt",
    confidence: float = 0.25,
):
    """Detect traffic signs in an image and save the annotated result."""
    image = Path(image_path)
    model_file = Path(model_path)

    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_file}. Train the model first."
        )

    model = YOLO(str(model_file))
    results = model.predict(
        source=str(image),
        conf=confidence,
        save=True,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect traffic signs in an image."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model", default="models/best.pt", help="Path to trained YOLO weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold"
    )
    args = parser.parse_args()

    predictions = predict_image(args.image, args.model, args.conf)

    for result in predictions:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            label = result.names[class_id]
            print(f"{label}: confidence={score:.3f}")
