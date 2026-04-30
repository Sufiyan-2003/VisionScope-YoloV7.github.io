import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels on image
    """
    # Convert to PIL for better text rendering
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = image.copy()
    
    draw = ImageDraw.Draw(image_pil)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]
        
        # Generate color based on class (consistent per class)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # Prepare label text
        label_text = f"{label}: {confidence:.2f}"
        
        # Calculate text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background for text
        draw.rectangle(
            [(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)],
            fill=color
        )
        
        # Draw text
        draw.text((x1 + 2, y1 - text_height - 3), label_text, fill=(255, 255, 255), font=font)
        
        # Add confidence score inside box if needed
        if confidence > 0.5:
            score_text = f"{confidence:.0%}"
            score_x = x1 + 5
            score_y = y1 + 5
            draw.text((score_x, score_y), score_text, fill=(255, 255, 0), font=font_small)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def display_detection_summary(detections):
    """
    Create a summary of detections as text
    """
    if not detections:
        return "No objects detected."
    
    summary = "Detection Summary:\n"
    summary += "=" * 30 + "\n"
    
    # Count objects by class
    class_counts = {}
    for detection in detections:
        label = detection["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    summary += f"Total objects: {len(detections)}\n\n"
    summary += "Objects found:\n"
    
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"  • {label}: {count}\n"
    
    summary += "\n" + "=" * 30 + "\n"
    summary += f"Highest confidence: {max(detections, key=lambda x: x['confidence'])['label']} ({max(detections, key=lambda x: x['confidence'])['confidence']:.2f})"
    
    return summary