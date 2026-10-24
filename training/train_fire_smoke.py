"""Reproducible fire/smoke YOLO training entrypoint (CPU-friendly smoke test)."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "datasets" / "fire_smoke.yaml"
OUT = ROOT / "runs" / "detect" / "fire_smoke"

def main():
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATA),
        epochs=10,
        imgsz=416,
        batch=2,
        workers=0,
        device="cpu",
        project=str(OUT.parent),
        name="fire_smoke",
        exist_ok=True,
        save=True,
    )
    print("Training complete. Output dir:", results.save_dir)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print("Best checkpoint:", best, "exists=", best.exists())
    # Evaluate the trained checkpoint explicitly
    trained = YOLO(str(best)) if best.exists() else model
    metrics = trained.val(data=str(DATA), imgsz=416, batch=2, device="cpu", workers=0)
    print("mAP50:", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("Class names:", trained.names)

if __name__ == "__main__":
    main()
