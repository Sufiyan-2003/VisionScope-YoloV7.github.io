import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def create_architecture_diagram():
    """
    Create a diagram explaining YOLOv7 architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("YOLOv7 Architecture - How It Works", fontsize=16, fontweight='bold', pad=20)
    
    # Define components
    components = [
        {"name": "Input Image\n640x640", "x": 0.5, "y": 2.5, "color": "#4CAF50"},
        {"name": "Backbone\n(E-ELAN)\nFeature Extraction", "x": 2.5, "y": 2.5, "color": "#2196F3"},
        {"name": "Neck\n(PANet)\nFeature Pyramid", "x": 5, "y": 2.5, "color": "#FF9800"},
        {"name": "Head\n(IDetect)\nPrediction", "x": 7.5, "y": 2.5, "color": "#9C27B0"},
        {"name": "Output\nBounding Boxes\n+ Classes", "x": 9.2, "y": 2.5, "color": "#f44336"}
    ]
    
    # Draw components
    for comp in components:
        rect = plt.Rectangle(
            (comp["x"] - 0.8, comp["y"] - 1.2), 1.6, 2.4,
            facecolor=comp["color"], alpha=0.8, edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(comp["x"], comp["y"], comp["name"], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', wrap=True)
    
    # Draw arrows
    arrows = [
        ((1.3, 2.5), (1.7, 2.5)),
        ((3.3, 2.5), (3.7, 2.5)),
        ((5.8, 2.5), (6.2, 2.5)),
        ((8.3, 2.5), (8.7, 2.5))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add explanation text at the bottom
    explanation = """How YOLOv7 Works:
1. Grid Division: Input image divided into S×S grid cells
2. Each cell predicts B bounding boxes and confidence scores
3. Backbone extracts features → Neck combines multi-scale features
4. Head predicts final boxes and class probabilities
5. NMS removes duplicate detections"""
    
    ax.text(5, 0.5, explanation, ha='center', va='bottom',
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0E0E0", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def create_feature_visualization(image, detections):
    """
    Create a simplified feature visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Detection result
    from .draw_utils import draw_detections
    result_img = draw_detections(image.copy(), detections)
    axes[0, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Detection Results")
    axes[0, 1].axis('off')
    
    # Confidence chart
    if detections:
        labels = [d['label'][:10] for d in detections[:10]]
        confidences = [d['confidence'] for d in detections[:10]]
        axes[0, 2].barh(labels, confidences, color='skyblue')
        axes[0, 2].set_xlabel('Confidence Score')
        axes[0, 2].set_title('Detection Confidence')
        axes[0, 2].set_xlim(0, 1)
    else:
        axes[0, 2].text(0.5, 0.5, "No detections", ha='center', va='center')
        axes[0, 2].set_title('No Objects Found')
    
    # Detection count pie chart
    if detections:
        class_counts = {}
        for d in detections:
            class_counts[d['label']] = class_counts.get(d['label'], 0) + 1
        
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Object Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, "No objects detected", ha='center', va='center')
        axes[1, 0].set_title('No Objects')
    
    # YOLO grid concept
    axes[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w = image.shape[:2]
    # Draw grid lines
    for i in range(1, 7):
        axes[1, 1].axhline(y=i * h/7, color='yellow', linewidth=0.5, alpha=0.5)
        axes[1, 1].axvline(x=i * w/7, color='yellow', linewidth=0.5, alpha=0.5)
    axes[1, 1].set_title('YOLO Grid Division (7×7)')
    axes[1, 1].axis('off')
    
    # Processing pipeline
    pipeline_steps = ['Input\nImage', 'Preprocess\n(640×640)', 'Feature\nExtraction', 
                      'Grid\nPrediction', 'NMS Filter', 'Output\nResult']
    steps_x = range(len(pipeline_steps))
    axes[1, 2].bar(steps_x, [5] * len(pipeline_steps), color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#f44336', '#4CAF50'])
    axes[1, 2].set_xticks(steps_x)
    axes[1, 2].set_xticklabels(pipeline_steps, rotation=45, ha='right')
    axes[1, 2].set_title('Detection Pipeline')
    axes[1, 2].set_ylim(0, 6)
    axes[1, 2].set_yticks([])
    
    plt.suptitle("YOLOv7 Detection Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def create_attention_heatmap(image, detections):
    """Create attention heatmap based on detection confidence"""
    heatmap = np.zeros((image.shape[0], image.shape[1]))
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        heatmap[y1:y2, x1:x2] += confidence
    
    # Normalize and apply colormap
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay on image
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    return overlay