import streamlit as st
import cv2
import numpy as np
import av
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

outfit_imgs = {
    "Outfit 1": cv2.imread('outfit1.png', cv2.IMREAD_UNCHANGED),
    "Outfit 2": cv2.imread('outfit2.png', cv2.IMREAD_UNCHANGED),
    "Outfit 3": cv2.imread('outfit3.png', cv2.IMREAD_UNCHANGED)
}

class SmartMirrorProcessor(VideoProcessorBase):
    def __init__(self):
        self.smooth_box = None
        self.alpha = 0.2
        self.frames_lost = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            curr_x, curr_y, curr_w, curr_h = faces[0]
            self.frames_lost = 0

            if self.smooth_box is None:
                self.smooth_box = [float(curr_x), float(curr_y), float(curr_w), float(curr_h)]
            else:
                self.smooth_box[0] = self.alpha * curr_x + (1 - self.alpha) * self.smooth_box[0]
                self.smooth_box[1] = self.alpha * curr_y + (1 - self.alpha) * self.smooth_box[1]
                self.smooth_box[2] = self.alpha * curr_w + (1 - self.alpha) * self.smooth_box[2]
                self.smooth_box[3] = self.alpha * curr_h + (1 - self.alpha) * self.smooth_box[3]
        else:
            self.frames_lost += 1
            if self.frames_lost > 10:
                self.smooth_box = None

        if self.smooth_box is not None:
            x, y, w, h = map(int, self.smooth_box)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            overlay_img = outfit_imgs.get(st.session_state.get("outfit_choice", "None"))
            if overlay_img is not None:
                try:
                    ow = int(w * 3.2)
                    oh = int(h * 3.5)
                    ox = int(x + (w / 2) - (ow / 2))
                    oy = int(y + h * 0.85)

                    resized_overlay = cv2.resize(overlay_img, (ow, oh))

                    y1, y2 = max(0, oy), min(img.shape[0], oy + oh)
                    x1, x2 = max(0, ox), min(img.shape[1], ox + ow)

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
                                img[y1:y2, x1:x2, c] = (
                                    alpha_s * overlay_crop[:, :, c] +
                                    alpha_l * img[y1:y2, x1:x2, c]
                                )
                except Exception:
                    pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col2:
    st.header("Live Mirror Feed")
    st.session_state["outfit_choice"] = outfit_choice
    webrtc_streamer(
        key="smart-mirror",
        video_processor_factory=SmartMirrorProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )
