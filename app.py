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
    
    # Camera Index Selector
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
        
        # Use CAP_DSHOW on Windows for reliable webcam access
        camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        if not camera.isOpened():
            st.error(f"Cannot open camera at index {cam_index}. Try changing Camera Index above or check if another app is using the webcam.")
        else:
            while run_camera:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to grab camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if overlay_img is not None:
                        try:
                            ow = int(w * 2.5)
                            oh = int(h * 3.0)
                            ox = int(x + (w / 2) - (ow / 2))
                            oy = int(y + h * 0.8)

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

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

            camera.release()
