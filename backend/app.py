from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import os
import uuid
from pathlib import Path
import sys

# ------------------ PATH SETUP ------------------

# Add project root (yolov7-main) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent))

# ------------------ IMPORTS ------------------

from model.yolo_model import YOLOv7Detector
from visualization.draw_utils import draw_detections, display_detection_summary
from visualization.explainability import create_architecture_diagram, create_feature_visualization

# ------------------ FLASK INIT ------------------

app = Flask(__name__)
CORS(app)

# ------------------ PATHS ------------------

BASE_DIR = Path(__file__).parent.parent
FRONTEND_PATH = BASE_DIR / "frontend"
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ------------------ LOAD MODEL ------------------

print("Initializing YOLOv7 Detector...")
detector = YOLOv7Detector(weights_path=str(BASE_DIR / "models" / "yolov7.pt"), device="cpu")
print("Model ready!")

# ------------------ ROUTES ------------------

# ✅ Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(FRONTEND_PATH, "index.html")

# ✅ Serve static files (CSS, JS)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_PATH, path)

# ✅ Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "YOLOv7 loaded"})

# ✅ Model info
@app.route('/model/info', methods=['GET'])
def get_model_info():
    info = detector.get_model_info()
    return jsonify(info)

# ✅ Architecture visualization
@app.route('/architecture', methods=['GET'])
def get_architecture():
    arch_image = create_architecture_diagram()
    return jsonify({"architecture_image": arch_image})

# ✅ Detection API
@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        conf_threshold = float(request.form.get('confidence_threshold', 0.25))
        iou_threshold = float(request.form.get('iou_threshold', 0.45))

        # Save uploaded image
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        image_path = UPLOAD_FOLDER / f"{unique_id}_original.{file_ext}"
        file.save(image_path)

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Detect
        detections = detector.detect(image, conf_threshold, iou_threshold)

        # Draw boxes
        result_image = draw_detections(image.copy(), detections)

        # Save result
        result_path = OUTPUT_FOLDER / f"{unique_id}_result.jpg"
        cv2.imwrite(str(result_path), result_image)

        # Summary
        summary = display_detection_summary(detections)

        # Feature visualization
        feature_viz = create_feature_visualization(image, detections)

        return jsonify({
            "success": True,
            "detections": detections,
            "total_objects": len(detections),
            "summary": summary,
            "result_image_url": f"/results/{result_path.name}",
            "feature_viz": feature_viz,
            "parameters": {
                "confidence_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Serve result images
@app.route('/results/<filename>', methods=['GET'])
def get_result_image(filename):
    result_path = OUTPUT_FOLDER / filename
    if result_path.exists():
        return send_file(str(result_path), mimetype='image/jpeg')
    return jsonify({"error": "Image not found"}), 404

# ------------------ RUN ------------------

if __name__ == '__main__':
    print("Starting VisionScope Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    app.run(host='0.0.0.0', port=5000, debug=True)