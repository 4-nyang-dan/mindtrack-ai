# ocr_pii/blur.py
import cv2

def union_boxes(boxes):
    if not boxes:
        return None
    x_coords = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
    y_coords = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def blur_area(image, box):
    x, y, w, h = [int(v) for v in box]
    if w <= 0 or h <= 0:
        return image
    
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return image
        
    roi_h, roi_w, _ = roi.shape
    pixel_size = 16
        
    temp_w = max(1, roi_w // pixel_size)
    temp_h = max(1, roi_h // pixel_size)
        
    temp = cv2.resize(roi, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(temp, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated_roi
    return image
