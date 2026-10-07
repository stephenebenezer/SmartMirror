import cv2
import numpy as np
import os

def overlay_transparent(background, overlay, x, y, target_w, target_h):
    """Overlay a PNG with transparency onto a camera frame."""
    bg_h, bg_w, _ = background.shape
    
    if target_w <= 0 or target_h <= 0:
        return background
    overlay_resized = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    if x >= bg_w or y >= bg_h or x + target_w <= 0 or y + target_h <= 0:
        return background
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + target_w), min(bg_h, y + target_h)
    
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)
    
    crop_overlay = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    crop_bg = background[y1:y2, x1:x2]
    
    if crop_overlay.shape[2] == 4:
        alpha = crop_overlay[:, :, 3] / 255.0
        alpha_3d = np.dstack([alpha] * 3)
        rgb_overlay = crop_overlay[:, :, :3]
        
        blended = (1.0 - alpha_3d) * crop_bg + alpha_3d * rgb_overlay
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
        
    return background

def main():
    xml_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_path):
        xml_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
    face_cascade = cv2.CascadeClassifier(xml_path)
    
    outfits = [
        ("Casual Tee", cv2.imread('outfit1.png', cv2.IMREAD_UNCHANGED)),
        ("Sportswear", cv2.imread('outfit2.png', cv2.IMREAD_UNCHANGED)),
        ("Formal Suit", cv2.imread('outfit3.png', cv2.IMREAD_UNCHANGED))
    ]
    current_index = 0
    y_offset = 0  # Dynamic offset for height fine-tuning
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("--- AI Smart Mirror Active ---")
    print("Press '1', '2', or '3' to switch outfits.")
    print("Press 'w' or 's' to adjust shirt height manually.")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lower threshold for low lighting detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # Fallback if low room lighting prevents detection
        if len(faces) == 0:
            h_f, w_f, _ = frame.shape
            # Default estimated face position in the center
            faces = [(int(w_f * 0.35), int(h_f * 0.15), int(w_f * 0.3), int(h_f * 0.3))]

        for (x, y, w, h) in faces:
            shirt_w = int(w * 3.0)
            shirt_h = int(h * 3.2)
            shirt_x = int(x - (shirt_w - w) / 2)
            shirt_y = int(y + h * 0.6) + y_offset # Added height fine-tuning
            
            current_name, current_img = outfits[current_index]
            if current_img is not None:
                frame = overlay_transparent(frame, current_img, shirt_x, shirt_y, shirt_w, shirt_h)

        # HUD Text
        current_name, _ = outfits[current_index]
        cv2.putText(frame, "AI Smart Mirror - Outfit Selector", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Outfit: {current_name} (Press 1,2,3 | W/S to move up/down)", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow('Smart Mirror Interface', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_index = 0
        elif key == ord('2'):
            current_index = 1
        elif key == ord('3'):
            current_index = 2
        elif key == ord('w'): # Move shirt up
            y_offset -= 10
        elif key == ord('s'): # Move shirt down
            y_offset += 10
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
