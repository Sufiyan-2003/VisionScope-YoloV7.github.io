import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Add YOLOv7 to path
yolo_path = Path(__file__).parent.parent / "yolov7"
sys.path.append(str(yolo_path))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class YOLOv7Detector:
    def __init__(self, weights_path="../models/yolov7.pt", device="cpu"):
        """
        Initialize YOLOv7 model
        """
        self.device = torch.device(device)
        self.weights_path = weights_path
        self.model = None
        self.names = None
        self.colors = None
        self.load_model()
        
    def load_model(self):
        """
        Load YOLOv7 model with weights
        """
        print(f"Loading model from {self.weights_path}...")
        self.model = attempt_load(self.weights_path, map_location=self.device)
        self.model.eval()
        
        # Get class names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.names), 3))
        
        print(f"Model loaded successfully. {len(self.names)} classes available.")
        
    def preprocess(self, image, img_size=640):
        """
        Preprocess image for YOLOv7 inference
        """
        # Resize and pad image
        img = letterbox(image, img_size, stride=32, auto=True)[0]
        
        # Convert HWC to CHW and normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # normalize to 0-1
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
    
    def detect(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run object detection on image
        """
        original_shape = image.shape
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)[0]
            
        # Apply NMS
        detections = non_max_suppression(
            predictions, 
            conf_thres=conf_threshold, 
            iou_thres=iou_threshold
        )[0]
        
        results = []
        
        if detections is not None and len(detections):
            # Scale boxes back to original image size
            detections[:, :4] = scale_coords(
                input_tensor.shape[2:], 
                detections[:, :4], 
                original_shape[:2]
            ).round()
            
            # Process each detection
            for detection in detections:
                x1, y1, x2, y2 = detection[:4].cpu().numpy().astype(int)
                confidence = float(detection[4])
                class_id = int(detection[5])
                label = self.names[class_id]
                
                results.append({
                    "label": str(label),
                    "confidence": float(confidence),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": int(class_id)
                })
                
        return results
    
    def get_model_info(self):
        """
        Return model architecture information
        """
        return {
            "name": "YOLOv7",
            "backbone": "E-ELAN (Extended Efficient Layer Aggregation Network)",
            "neck": "PANet (Path Aggregation Network)",
            "head": "IDetect (Integrated Detection Head)",
            "parameters": "37.2M",
            "classes": len(self.names),
            "input_size": "640x640"
        }