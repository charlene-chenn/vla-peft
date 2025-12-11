from pathlib import Path

from ultralytics import YOLO


class YOLOv8Detector:
    """
    A wrapper class for the YOLOv8 model from Ultralytics.

    This class provides a structured interface for training, validating, and
    running inference with a YOLOv8 model, tailored for the GeoGuesser project.
    It standardizes interactions with the underlying `ultralytics` library and
    is designed to be used as a component in a larger pipeline.

    Attributes:
        model (YOLO): The underlying Ultralytics YOLO model instance.
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initializes the YOLOv8Detector.

        To train from scratch, pass a model configuration file (e.g., 'yolov8n.yaml').
        To fine-tune, pass a pre-trained model file (e.g., 'yolov8n.pt').
        To load a custom-trained model for inference, pass the path to your
        trained weights file (e.g., 'runs/detect/train1/weights/best.pt').

        Args:
            model_path (str): The path to the model file or configuration yaml.
        """
        self.model = YOLO(model_path)
        print(f"YOLOv8Detector initialized with model: {model_path}")

    def train(self, data_yaml: str, epochs: int, batch_size: int, resume: bool = False, **kwargs):
        """
        Handles the training of the model.

        This method leverages Ultralytics' built-in training loop, which
        automatically handles checkpointing, logging, and saving of metrics.

        To resume a crashed or stopped training run, set `resume=True`. The
        library will automatically find the last checkpoint.

        All results, including model weights, metrics, and graphs, are saved
        to a unique directory in `runs/detect/`.

        Args:
            data_yaml (str): Path to the dataset .yaml file.
            epochs (int): Number of training epochs.
            batch_size (int): Number of images per batch.
            resume (bool): If True, resumes training from the last checkpoint.
            **kwargs: Additional arguments to pass to the underlying train method.
                      e.g., imgsz=640, project='runs/detect', name='my_experiment'

        Returns:
            The results object from the Ultralytics training run.
        """
        print(f"Starting training with data: {data_yaml}...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            resume=resume,
            **kwargs
        )
        print("Training complete.")
        print(f"Model and results saved to: {results.save_dir}")
        return results

    def predict(self, image, conf: float = 0.5):
        """
        Performs inference on a given image and returns structured data.

        Args:
            image: The input image. Can be a file path, a PIL Image, or a
                   NumPy array.
            conf (float): The confidence threshold for detections.

        Returns:
            dict: A dictionary mapping class names to a list of instances.
                  Each instance is a dictionary containing confidence, a
                  bounding box for cropping, and the scale of the box.
                  Example:
                  {
                      "traffic_light": [
                          {"confidence": 0.95, "bbox_crop": [100, 150, 120, 180], "scale": 600},
                          ...
                      ],
                      "utility_pole": [
                          {"confidence": 0.88, "bbox_crop": [250, 100, 260, 250], "scale": 1500}
                      ]
                  }
        """
        results = self.model.predict(image, conf=conf, verbose=False)

        # Results is a list, but we process one image at a time for this method
        result = results[0]

        # Use a defaultdict to easily append to lists
        from collections import defaultdict
        output = defaultdict(list)

        # Get class names from the model
        class_names = result.names

        for box in result.boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Get confidence score
            confidence = float(box.conf[0])

            # Get bounding box for cropping (x1, y1, x2, y2)
            bbox_crop = [int(coord) for coord in box.xyxy[0]]
            x1, y1, x2, y2 = bbox_crop

            # Calculate scale (area of the bounding box)
            scale = (x2 - x1) * (y2 - y1)

            output[class_name].append({
                "confidence": confidence,
                "bbox_crop": bbox_crop,
                "scale": scale
            })

        return dict(output)

    def validate(self, data_yaml: str = None, split: str = 'test'):
        """
        Validates the model on a specified dataset split.

        Args:
            data_yaml (str, optional): Path to the dataset .yaml file. If None,
                                       the one from the last training run is used.
            split (str): The dataset split to use for validation ('val' or 'test').

        Returns:
            The metrics object from the Ultralytics validation run.
        """
        print(f"Running validation on the '{split}' split...")
        metrics = self.model.val(data=data_yaml, split=split)
        print("Validation complete.")
        return metrics

    def visualize_performance(self, project_dir: str = 'runs/detect'):
        """
        Finds the latest training run and prints the paths to key
        performance metric files.

        The Ultralytics library automatically generates several files with
        graphs and data. This method provides easy access to them.

        Args:
            project_dir (str): The parent directory for training runs.
        """
        latest_run_dir = self._get_latest_run_dir(project_dir)

        if not latest_run_dir:
            print(f"No training runs found in '{project_dir}'.")
            return

        print(f"--- Performance Metrics for Latest Run ---")
        print(f"Results directory: {latest_run_dir}\n")

        key_files = {
            "results.csv": "CSV file with per-epoch metrics (loss, mAP, etc.).",
            "results.png": "Graph showing training and validation metrics over epochs.",
            "confusion_matrix.png": "Confusion matrix for the validation set.",
            "PR_curve.png": "Precision-Recall curve.",
        }

        print("You can find the following key performance files in the directory:")
        for filename, description in key_files.items():
            file_path = latest_run_dir / filename
            if file_path.exists():
                print(f"  - {filename}: {description}")
                print(f"    Path: {file_path}")

    def _get_latest_run_dir(self, project_dir: str = 'runs/detect') -> Path | None:
        """
        Finds the latest 'train' directory (e.g., train, train2, train3, ...).
        """
        project_path = Path(project_dir)
        if not project_path.exists():
            return None

        train_dirs = list(project_path.glob('train*'))
        if not train_dirs:
            return None

        # Find the one with the highest number, or 'train' if no number
        latest_dir = max(train_dirs, key=lambda p: int(''.join(filter(str.isdigit, p.name)) or 0))
        return latest_dir


if __name__ == '__main__':
    # This is for demonstration purposes.
    # We will create a more complete main.py entrypoint later.
    print("YOLOv8Detector class created. Ready to be implemented.")

    # Example of how it will be initialized
    # detector = YOLOv8Detector('yolov8n.pt')
