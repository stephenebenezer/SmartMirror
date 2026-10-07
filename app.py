import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
    run_camera = st.checkbox("Turn On Camera", value=True)
    
    if run_camera:
        # Load Face Cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Load Outfits if selected
        overlay_img = None
        if outfit_choice == "Outfit 1":
            overlay_img = cv2.imread('outfit1.png', cv2.IMREAD_UNCHANGED)
        elif outfit_choice == "Outfit 2":
            overlay_img = cv2.imread('outfit2.png', cv2.IMREAD_UNCHANGED)
        elif outfit_choice == "Outfit 3":
            overlay_img = cv2.imread('outfit3.png', cv2.IMREAD_UNCHANGED)

        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            # Mirror frame horizontally for standard mirror feel
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Overlay logic if an outfit is selected
                if overlay_img is not None:
                    try:
                        # Position outfit below neck based on detected face bounding box
                        ow = int(w * 2.2)
                        oh = int(h * 2.8)
                        ox = int(x + (w / 2) - (ow / 2))
                        oy = int(y + h * 0.9)

                        if ox > 0 and oy > 0 and ox + ow < frame.shape[1] and oy + oh < frame.shape[0]:
                            resized_overlay = cv2.resize(overlay_img, (ow, oh))
                            alpha_s = resized_overlay[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s

                            for c in range(0, 3):
                                frame[oy:oy+oh, ox:ox+ow, c] = (
                                    alpha_s * resized_overlay[:, :, c] +
                                    alpha_l * frame[oy:oy+oh, ox:ox+ow, c]
                                )
                    except Exception:
                        pass

            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        camera.release()
