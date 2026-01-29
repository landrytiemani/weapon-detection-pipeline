from ultralytics import RTDETR
from ultralytics import YOLO

class RTDETRDetector:
    def __init__(self, config):
        """
        Initializes the RT-DETR model.
        Args:
            config (dict): A dictionary containing model configuration.
        """
        self.model = RTDETR(config['model_path'])
        print(f"OBJECT DETECTOR: Loaded RT-DETR from: {config['model_path']}")

        # self.model = YOLO(config['model_path'])
        # print("model>summary: ", self.model.model.info(verbose=True) )
        self.config = config

    def detect(self, frame):
        """
        Performs weapon detection on a single frame.
        Args:
            frame: The input frame.
        Returns:
            The detection results.
        """
        results = self.model(frame, conf=self.config['confidence_threshold'])
        return results