import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

outfit_imgs = {
    "Outfit 1": cv2.imread(os.path.join(BASE_DIR, 'outfit1.png'), cv2.IMREAD_UNCHANGED),
    "Outfit 2": cv2.imread(os.path.join(BASE_DIR, 'outfit2.png'), cv2.IMREAD_UNCHANGED),
    "Outfit 3": cv2.imread(os.path.join(BASE_DIR, 'outfit3.png'), cv2.IMREAD_UNCHANGED)
}

cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(cascade_path):
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)
    except Exception:
        pass

with col2:
    st.header("Live Mirror Feed")
    
    # Store camera photo in session_state so it doesn't clear on outfit selection
    img_file_buffer = st.camera_input("Take a photo to preview virtual outfit")

    if img_file_buffer is not None:
        st.session_state["saved_photo"] = img_file_buffer.getvalue()

    if "saved_photo" in st.session_state:
        bytes_data = st.session_state["saved_photo"]
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2_img = cv2.flip(cv2_img, 1)

        img_h, img_w, _ = cv2_img.shape

        face_cascade = None
        if os.path.exists(cascade_path):
            try:
                face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                face_cascade = None

        faces = []
        if face_cascade is not None and not face_cascade.empty():
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            (x, y, w, h) = faces[0]
            cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ow = int(w * 3.2)
            oh = int(h * 3.5)
            ox = int(x + (w / 2) - (ow / 2))
            oy = int(y + h * 0.85)
        else:
            # Automatic center chest overlay fallback if face detection drops
            ow = int(img_w * 0.7)
            oh = int(img_h * 0.75)
            ox = int((img_w - ow) / 2)
            oy = int(img_h * 0.3)

        overlay_img = outfit_imgs.get(outfit_choice)
        if overlay_img is not None:
            try:
                resized_overlay = cv2.resize(overlay_img, (ow, oh))

                y1, y2 = max(0, oy), min(img_h, oy + oh)
                x1, x2 = max(0, ox), min(img_w, ox + ow)

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
                            cv2_img[y1:y2, x1:x2, c] = (
                                alpha_s * overlay_crop[:, :, c] +
                                alpha_l * cv2_img[y1:y2, x1:x2, c]
                            )
            except Exception:
                pass

        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        st.image(cv2_img_rgb, caption=f"Active Outfit: {outfit_choice}", use_container_width=True)
