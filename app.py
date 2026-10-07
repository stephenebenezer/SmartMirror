import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="AI Smart Mirror", layout="wide")

st.title("AI-Enabled Smart Mirror")
st.subheader(datetime.now().strftime("%A, %B %d, %Y - %I:%M %p"))

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Outfit Selection")
    outfit_choice = st.radio(
        "Choose an Outfit:",
        ["None", "Outfit 1", "Outfit 2", "Outfit 3"]
    )
    
    st.markdown("---")
    st.header("Mirror Widgets")
    st.write("• **Weather:** 29°C / Partly Cloudy")
    st.write("• **Schedule:** Science Exhibition Demo")

with col2:
    st.header("Live Mirror Feed")
    cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    run_camera = st.checkbox("Turn On Camera", value=True)
    
    if run_camera:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        overlay_img = None
        if outfit_choice == "Outfit 1":
            overlay_img = cv2.imread('outfit1.png', cv2.IMREAD_UNCHANGED)
        elif outfit_choice == "Outfit 2":
            overlay_img = cv2.imread('outfit2.png', cv2.IMREAD_UNCHANGED)
        elif outfit_choice == "Outfit 3":
            overlay_img = cv2.imread('outfit3.png', cv2.IMREAD_UNCHANGED)

        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        if not camera.isOpened():
            st.error(f"Cannot open camera at index {cam_index}.")
        else:
            # Persistent bounding box & frame loss counter for smoothing
            smooth_box = None  # Holds smoothed (x, y, w, h)
            alpha = 0.2        # Smoothing weight (lower = smoother)
            frames_lost = 0    # Keep last position for up to 10 dropped frames

            while run_camera:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to grab camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=4, 
                    minSize=(60, 60)
                )

                if len(faces) > 0:
                    # Pick largest face
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    curr_x, curr_y, curr_w, curr_h = faces[0]
                    frames_lost = 0

                    if smooth_box is None:
                        smooth_box = [float(curr_x), float(curr_y), float(curr_w), float(curr_h)]
                    else:
                        # EMA Smoothing
                        smooth_box[0] = alpha * curr_x + (1 - alpha) * smooth_box[0]
                        smooth_box[1] = alpha * curr_y + (1 - alpha) * smooth_box[1]
                        smooth_box[2] = alpha * curr_w + (1 - alpha) * smooth_box[2]
                        smooth_box[3] = alpha * curr_h + (1 - alpha) * smooth_box[3]
                else:
                    frames_lost += 1
                    if frames_lost > 10:
                        smooth_box = None  # Clear if lost for too long

                if smooth_box is not None:
                    x, y, w, h = map(int, smooth_box)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if overlay_img is not None:
                        try:
                            ow = int(w * 3.2)
                            oh = int(h * 3.5)
                            ox = int(x + (w / 2) - (ow / 2))
                            oy = int(y + h * 0.85)

                            resized_overlay = cv2.resize(overlay_img, (ow, oh))

                            y1, y2 = max(0, oy), min(frame.shape[0], oy + oh)
                            x1, x2 = max(0, ox), min(frame.shape[1], ox + ow)

                            overlay_y1 = max(0, -oy)
                            overlay_y2 = overlay_y1 + (y2 - y1)
                            overlay_x1 = max(0, -ox)
                            overlay_x2 = overlay_x1 + (x2 - x1)

                            if y1 < y2 and x1 < x2:
                                overlay_crop = resized_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
                                
                                if overlay_crop.shape[2] == 4:
                                    alpha_s = overlay_crop[:, :, 3] / 255.0
                                    alpha_l = 1.0 - alpha_s

                                    for c in range(0, 3):
                                        frame[y1:y2, x1:x2, c] = (
                                            alpha_s * overlay_crop[:, :, c] +
                                            alpha_l * frame[y1:y2, x1:x2, c]
                                        )
                                else:
                                    frame[y1:y2, x1:x2] = overlay_crop[:, :, :3]
                        except Exception:
                            pass
                elif overlay_img is not None:
                    cv2.putText(
                        frame, 
                        "Position face inside frame to view outfit", 
                        (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), 
                        2
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

            camera.release()
