from ultralytics import YOLO
from pathlib import Path

current = Path(__file__).parent
# Load a model
model = YOLO("yolov8s.pt").to("cuda")  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    # Train the model
    results = model.train(
        data=str(current / "yolo_datasets/fruit.yaml"),
        epochs=150,
        workers=0,
        device="cuda",
        imgsz=640,
        project="fruit3",
        save_period=10,
        augment=True,
    )
    model.val()
    model.save("fruit3.pt")
