import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Smart Mirror", layout="wide")

# Header Section
st.title("Smart Mirror Dashboard")
st.subheader(datetime.now().strftime("%A, %B %d, %Y - %I:%M %p"))

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.header("Widget Area")
    st.write("- Weather: 72F / Sunny")
    st.write("- Calendar: No events today")

with col2:
    st.header("Camera Feed")
    run_camera = st.checkbox("Turn On Camera")
    
    if run_camera:
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break
            
            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
        camera.release()
