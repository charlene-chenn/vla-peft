import argparse
import pprint
import sys
from pathlib import Path

from yolo_detector.model import YOLOv8Detector


def main():
    """
    Main entrypoint for the YOLOv8 detector module.
    Provides a CLI for training, validating, and predicting.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Detector Module CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train a YOLOv8 model.")
    parser_train.add_argument("--model", type=str, default="yolov8n.yaml", help="Path to model config for scratch training (e.g., yolov8n.yaml) or weights for fine-tuning (e.g., yolov8n.pt).")
    parser_train.add_argument("--data", type=str, default="configs/vistas_yolo.yaml", help="Path to the dataset YAML file.")
    parser_train.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser_train.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser_train.add_argument("--workers", type=int, default=8, help="Number of worker threads for data loading.")
    parser_train.add_argument("--cache", type=str, default=None, choices=['ram', 'disk'], help="Cache dataset to RAM or disk for faster training.")
    parser_train.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    parser_train.add_argument("--name", type=str, default="vistas_yolo_run", help="Name of the training run.")
    parser_train.add_argument("--device", type=str, default="auto", help="Device to run training on, e.g., 'cpu', '0', 'mps'.")

    # --- Validate Command ---
    parser_validate = subparsers.add_parser("validate", help="Validate a trained YOLOv8 model.")
    parser_validate.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (e.g., runs/detect/exp/weights/best.pt).")
    parser_validate.add_argument("--data", type=str, default="configs/vistas_yolo.yaml", help="Path to the dataset YAML file.")
    parser_validate.add_argument("--split", type=str, default="test", choices=['val', 'test'], help="Dataset split to use for validation.")

    # --- Visualize Command ---
    parser_visualize = subparsers.add_parser("visualize", help="Show paths to performance metrics of a training run.")
    parser_visualize.add_argument("--run", type=str, default=None, help="Specific run directory to visualize (e.g., runs/detect/vistas_yolo_run). If not provided, the latest run will be used.")

    # --- Predict Command ---
    parser_predict = subparsers.add_parser("predict", help="Run inference with a trained YOLOv8 model.")
    parser_predict.add_argument("--weights", type=str, required=True, help="Path to the trained model weights.")
    parser_predict.add_argument("--source", type=str, required=True, help="Path to the source image or directory.")
    parser_predict.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for predictions.")

    args = parser.parse_args()

    if args.command == "train":
        # Initialize for training
        detector = YOLOv8Detector(model_path=args.model)
        detector.train(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            workers=args.workers,
            cache=args.cache,
            resume=args.resume,
            project='runs/detect',
            name=args.name,
            device=args.device
        )
        # After training, show where to find results
        detector.visualize_performance(project_dir=f'runs/detect/{args.name}')

    elif args.command == "validate":
        # Initialize with trained weights
        detector = YOLOv8Detector(model_path=args.weights)
        detector.validate(data_yaml=args.data, split=args.split)

    elif args.command == "visualize":
        # Instantiate a dummy detector to use the helper method
        detector = YOLOv8Detector()
        run_dir = args.run or 'runs/detect'
        detector.visualize_performance(project_dir=run_dir)

    elif args.command == "predict":
        # Initialize with trained weights
        detector = YOLOv8Detector(model_path=args.weights)
        predictions = detector.predict(image=args.source, conf=args.conf)

        print(f"\n--- Predictions for {args.source} ---")
        pprint.pprint(predictions)
        print("------------------------------------")


if __name__ == "__main__":
    main()
